# Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.26 | Core numerical operations — matrices, linear algebra |
| scipy | >=1.12 | Statistical distributions (future use), optimization |
| pytest | >=7.0 | Test framework |

No external active inference libraries used. Built from scratch to keep the implementation minimal and fully understood.

## Why Not pymdp?

pymdp (v0.0.7.1) is the only Python active inference library. It was evaluated and rejected because:
- Discrete state spaces only (same limitation we have, but no clear path to extend)
- API is complex and not well-documented
- No production deployments
- Building from scratch gives full control and better understanding of the math
