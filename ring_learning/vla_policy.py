"""Task-specific small image/text-conditioned Cartesian action-chunk model.

This is a hierarchical minimal VLA, not a pretrained general-purpose VLA.
Learned foreground masks locate two objects under a fixed, declared simulation
camera. Learned word embeddings condition complete XYZ/gripper waypoint chunks.
The downstream servo may use robot proprioception, but no object ground truth.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import torch
from torch import nn


SCHEMA_VERSION = 1
N_WAYPOINTS = 23
FEATURE_DIM = 10
CAMERA_CENTRE_XY = (.47, 0.)
CAMERA_SPAN_M = .56
INSTRUCTION_GROUPS = {
    "inside": ("put the ring inside the bearing", "place the ring inside the bearing",
               "insert the ring into the bearing"),
    "left": ("put the ring left of the bearing", "place the ring to the left of the bearing",
             "move the ring to the left of the bearing"),
    "right": ("put the ring right of the bearing", "place the ring to the right of the bearing",
              "move the ring to the right of the bearing"),
}
PRIMARY_INSTRUCTIONS = {name: phrases[0] for name, phrases in INSTRUCTION_GROUPS.items()}
ALLOWED_INSTRUCTIONS = tuple(phrase for phrases in INSTRUCTION_GROUPS.values() for phrase in phrases)
VOCABULARY = ("<pad>",)+tuple(sorted({word for phrase in ALLOWED_INSTRUCTIONS for word in phrase.split()}))
TOKEN_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}


def encode_text(instructions, *, device=None):
    """Encode the nine supported English phrases as actual padded word tokens.

    The registry constrains the supported grammar; it does not provide task IDs
    to the network or select an action branch. Untrained commands fail explicitly.
    """
    if isinstance(instructions, str) or not isinstance(instructions, (list, tuple)) or not instructions:
        raise ValueError("instructions must be a nonempty list of supported text instructions")
    rows = []
    for instruction in instructions:
        if not isinstance(instruction, str):
            raise ValueError("instruction must be supported English text")
        normalized = re.sub(r"\s+", " ", instruction.strip().lower()).rstrip(".!?")
        if normalized not in ALLOWED_INSTRUCTIONS:
            raise ValueError(f"Unsupported instruction: {instruction!r}")
        rows.append([TOKEN_TO_ID[word] for word in normalized.split()])
    tokens = torch.zeros((len(rows), max(map(len, rows))), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        tokens[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    return tokens


def scene_feature_basis(coordinates):
    """Declared geometric basis from predicted ring/bearing world XY, in metres."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 4 or not torch.isfinite(coordinates).all():
        raise ValueError("scene coordinates must be a finite N x 4 tensor")
    direction = coordinates[:, 2:4]-coordinates[:, :2]
    unit = direction/torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-6)
    ux, uy = unit[:, 0:1], unit[:, 1:2]
    return torch.cat((coordinates, torch.ones_like(ux), ux, uy, ux.square(), ux*uy, uy.square()), dim=1)


class TinyVLA(nn.Module):
    """Learned pixel masks + word-conditioned matrix → 23 XYZ/grip waypoints."""

    feature_dim = FEATURE_DIM

    def __init__(self, n_waypoints=N_WAYPOINTS, *, embedding_dim=16,
                 camera_centre_xy=CAMERA_CENTRE_XY, camera_span_m=CAMERA_SPAN_M,
                 centroid_temperature=1.):
        super().__init__()
        if (not isinstance(n_waypoints, int) or isinstance(n_waypoints, bool) or n_waypoints < 1
                or not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim < 1):
            raise ValueError("waypoint and embedding dimensions must be positive integers")
        centre = torch.as_tensor(camera_centre_xy, dtype=torch.float32)
        if (centre.shape != (2,) or not torch.isfinite(centre).all()
                or not 0 < float(camera_span_m) < float("inf")
                or not 0 < float(centroid_temperature) < float("inf")):
            raise ValueError("camera calibration and centroid temperature must be finite and valid")
        self.n_waypoints = n_waypoints
        self.embedding_dim = embedding_dim
        self.camera_span_m = float(camera_span_m)
        self.centroid_temperature = float(centroid_temperature)
        self.register_buffer("camera_centre_xy", centre.clone())
        self.vision = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(16, 2, 1))
        self.token_embedding = nn.Embedding(len(VOCABULARY), embedding_dim, padding_idx=0)
        # Nonlinearity is necessary here: a single affine map of mean word
        # embeddings cannot give equal left/right offsets to aliases of
        # different lengths, because one distinguishing token is diluted.
        self.coefficient_head = nn.Sequential(nn.Linear(embedding_dim, 32), nn.ReLU(),
                                               nn.Linear(32, n_waypoints*4*FEATURE_DIM))

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def decoder_parameters(self):
        return list(self.token_embedding.parameters())+list(self.coefficient_head.parameters())

    def text_features(self, tokens):
        if (tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[0] < 1
                or tokens.shape[1] < 1 or bool((tokens < 0).any())
                or bool((tokens >= len(VOCABULARY)).any())):
            raise ValueError("text tokens must be a valid nonempty padded word-token batch")
        mask = tokens.ne(0)
        if bool((mask.sum(dim=1) == 0).any()):
            raise ValueError("text tokens cannot contain only padding")
        embeddings = self.token_embedding(tokens)
        return (embeddings*mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1, keepdim=True)

    def scene_features_from_logits(self, logits):
        """Convert learned spatial attention into fixed-camera metric features."""
        if (logits.ndim != 4 or logits.shape[1] != 2 or min(logits.shape[2:]) < 2
                or not logits.is_floating_point() or not torch.isfinite(logits).all()):
            raise ValueError("mask logits must be finite N x 2 x H x W")
        height, width = logits.shape[2:]
        foreground = torch.sigmoid(logits/self.centroid_temperature).square()
        # Squaring suppresses the many low-probability background pixels while
        # saturated foreground pixels retain almost uniform mass. A softmax
        # would instead concentrate on the brightest part of a metal object.
        mass = foreground.sum(dim=(2, 3), keepdim=True)
        weights = foreground/mass.clamp_min(torch.finfo(logits.dtype).tiny)
        uniform = torch.full_like(weights, 1/(height*width))
        weights = torch.where(mass > torch.finfo(logits.dtype).tiny, weights, uniform)
        columns = (torch.arange(width, device=logits.device, dtype=logits.dtype)+.5)/width
        rows = (torch.arange(height, device=logits.device, dtype=logits.dtype)+.5)/height
        x = (weights*columns.view(1, 1, 1, width)).sum(dim=(2, 3))
        y = (weights*rows.view(1, 1, height, 1)).sum(dim=(2, 3))
        world_x = self.camera_centre_xy[0]+(x-.5)*self.camera_span_m
        world_y = self.camera_centre_xy[1]+(.5-y)*self.camera_span_m
        coordinates = torch.stack((world_x[:, 0], world_y[:, 0], world_x[:, 1], world_y[:, 1]), dim=1)
        return scene_feature_basis(coordinates)

    def scene_features(self, images):
        if (images.ndim != 4 or images.shape[1] != 3 or min(images.shape[2:]) < 2
                or not images.is_floating_point() or not torch.isfinite(images).all()
                or bool((images < 0).any()) or bool((images > 1).any())):
            raise ValueError("images must be finite floating N x 3 x H x W RGB in [0, 1]")
        logits = self.vision(images)
        return self.scene_features_from_logits(logits), logits

    def predict_from_scene_features(self, features, tokens):
        """Training helper: prefit language/action decoder on labeled geometry.

        Deployment obtains these features only from the learned RGB masks.
        """
        if (features.ndim != 2 or features.shape[1] != FEATURE_DIM
                or not torch.isfinite(features).all() or features.shape[0] != tokens.shape[0]):
            raise ValueError("scene features must be finite N x 10 and match text batch")
        coefficients = self.coefficient_head(self.text_features(tokens)).reshape(-1, self.n_waypoints, 4, FEATURE_DIM)
        return torch.einsum("nkaf,nf->nka", coefficients, features)

    def forward(self, images, tokens):
        features, logits = self.scene_features(images)
        return self.predict_from_scene_features(features, tokens), logits

    def configuration(self):
        return dict(n_waypoints=self.n_waypoints, embedding_dim=self.embedding_dim,
                    camera_centre_xy=self.camera_centre_xy.detach().cpu().tolist(),
                    camera_span_m=self.camera_span_m, centroid_temperature=self.centroid_temperature)


def policy_source_sha256():
    contents = Path(__file__).read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


def save_vla_policy(path, model, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema_version=SCHEMA_VERSION, source_sha256=policy_source_sha256(),
                   vocabulary=list(VOCABULARY), allowed_instructions=list(ALLOWED_INSTRUCTIONS),
                   configuration=model.configuration(),
                   state_dict={name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
                   metadata=metadata)
    torch.save(payload, path)


def load_vla_policy(path, *, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported VLA checkpoint schema")
    if payload.get("source_sha256") != policy_source_sha256():
        raise ValueError("VLA checkpoint source differs from current policy source")
    if payload.get("vocabulary") != list(VOCABULARY) or payload.get("allowed_instructions") != list(ALLOWED_INSTRUCTIONS):
        raise ValueError("VLA checkpoint language registry differs from current policy source")
    model = TinyVLA(**payload["configuration"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError("VLA checkpoint contains nonfinite parameters")
    return model.to(device).eval(), payload["metadata"]
