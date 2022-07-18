# How did we get here
-------------------
Whilst reading through Sedgewick's algorithms, early on one discovers a humble strip of code for traversing or reversing (or whatever you wish) to a double array. I've always just used builtins to achieve this functionality, and thus, I resolved to once and for all understand what was actually happening in-depth.

A few weeks later, it's a fully fledged effort to implement in Python fundamental linear algebra datatypes and functions for help with visualizing (and solving) problems from David Lay's Linear Algebra and Its Applications 5th Edition.

Designed less for sake of wonderful typing and more for having "concrete" instances of fundamental objects described in linear algebra classes.

## Things Implemented
* Basic matrix objects
* Simple matrix operations
    * addition
    * traversals
    * rotations
* Augmented Matrices
    * with types for Coefficient and Constant matrices
* Gaussian Elimination (Row Echelon Form)
    * Solving with Back Substitution
* Gauss-Jordan (Reduced Row Echelon Form)
    * Trivial solving algorithm
