# A simple python implementation of Torczon algorithm

Torczon algorithm is a derivative free algorithm initially designed for unconstrained nonlinear optimization problems. 
Torczon algorithm is a version of the simplex algorithm (not that used in Linear Programming problems) which avoids the 
Collapse of the polygone of solutions that is iteratively updated through the iterations.

This is a modified version of the Torczon algorithm that incorporates:

- Explicitly handling of hard box constraints on the decision variables
- Penalty-based handling of other (non box-like) constraints

The main appealing feature of this family of algorithm lies in the fact that the cost function and the constraints need 
not to be differentiable. The counterpart is that this algorithm is not fitted to very high dimensional problems. 

I personally use it intensively in solving real-life problems arising in **Parameterized Nonlinear Model Predictive Control** 
problems. 
