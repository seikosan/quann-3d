"""Set-valued complaint narratives encoded with the canonical cloud core."""

from torch import nn

from cloudlm.model import CloudEncoderND, WordToND


class ComplaintCloud(nn.Module):
    """Embedding → unordered word cloud → geometric state → reader and issue head."""

    def __init__(
        self, vocab_size, d_model=128, d_cloud_dim=32, d_cloud=128,
        hidden=128, d_z=32, n_issue=None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.word_to_nd = WordToND(d_model=d_model, d_cloud_dim=d_cloud_dim, hidden=hidden)
        self.cloud_encoder = CloudEncoderND(
            d_cloud_dim=d_cloud_dim, d_cloud=d_cloud, hidden=hidden,
        )
        self.reader = nn.Sequential(
            nn.Linear(d_cloud, hidden), nn.GELU(), nn.Linear(hidden, d_z),
        )
        self.issue_head = nn.Linear(d_z, n_issue) if n_issue else None

    def forward(self, tokens, mask):
        points = self.word_to_nd(self.embedding(tokens))
        cloud_state = self.cloud_encoder(points, mask.bool())
        z = self.reader(cloud_state)
        if self.issue_head is None:
            return cloud_state, z
        return cloud_state, z, self.issue_head(z)
