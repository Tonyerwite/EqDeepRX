# EqDeepRx Open-Source Search

The requested external search was performed with the `agent-reach` GitHub/web route on 2026-09-09/10.

- Direct GitHub repository check: [Tonyerwite/EqDeepRX](https://github.com/Tonyerwite/EqDeepRX) responded successfully, while `git ls-remote https://github.com/Tonyerwite/EqDeepRX.git` returned no refs. It is therefore treated as the requested empty destination repository, not as an existing implementation to copy.
- GitHub CLI repository/code search was attempted but the local CLI had no authenticated GitHub session. The unauthenticated GitHub API/search endpoints returned rate-limit responses.
- General web search for `EqDeepRx GitHub implementation code` and the full paper title returned no directly reusable EqDeepRx source tree. Search pages either contained no matching project or were blocked by search-provider rate limits.
- No public repository was found that implements the paper's DenoiseNN + parallel RZF/LMMSE + shared per-layer DetectorNN/DemapperNN pipeline. 因此没有发现可直接复用的 EqDeepRx 开源实现，本项目按论文和现有 DeepRx 链路重写。

The search limitation is recorded explicitly: absence of a search hit is not proof that no private or unindexed implementation exists. The new code is therefore original integration code with the supplied DeepRx project used only as an architectural and tensor-layout reference.

