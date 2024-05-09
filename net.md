```mermaid
flowchart TB
  subgraph 2817608955600[0:@ddf:pla_OR: lay]
    input -- pla_OR\n  (_x, Type){'Type': 'OR'}--> 2819129246048((0:pla_OR))
    input -- pla_NAND\n  (_x, Type){'Type': 'NAND'}--> 2819129245904((1:pla_NAND))
  end
  2819129246048((0:pla_OR)) -- pla_AND\n  (_x, Type){'Type': 'AND'}--> 2819129252528((1:pla_AND))
  2819129245904((1:pla_NAND)) -- pla_AND\n  (_x, Type){'Type': 'AND'}--> 2819129252528((1:pla_AND))
  2819129252528((1:pla_AND)) --> output
```