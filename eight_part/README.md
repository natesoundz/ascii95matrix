# Eight-part expert-path branch

This branch evaluates a conditional-path variant of the eight-part architecture against weights produced by the repository ASCII95 compiler.

Execution contract:

1. compile ASCII95 evidence into frozen weights;
2. retain one persistent numerical state;
3. invoke attention only when visible relations are available;
4. route through a subset of the compiled FFN correction field as experts;
5. do not normalize between expert transformations;
6. normalize once at path exit;
7. decode against the frozen ASCII95 token geometry.

No optimizer is introduced by the expert-path runtime.
