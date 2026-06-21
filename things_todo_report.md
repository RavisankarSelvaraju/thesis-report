

## Background
- write about the inefk's propagate step, the state transition matrix and noise matrix and how they work

## Methodology


\section{Slip-Aware Measurement Rejection}
Explain the idea of rejecting inconsistent wheel measurements.
Describe the use of innovation uncertainty / Mahalanobis distance at concept level.
State that wheel updates are rejected when the measurement is inconsistent with the filter prediction.

\section{Contact-State Estimation}
Explain how the contact state of spokes is inferred.
Keep it algorithmic, not code-specific.
Describe how innovation consistency is used to estimate likely contacting spokes.

\section{Assumptions and Limitations of the Method}
Mention assumptions such as:
known kinematic model, known spoke geometry, encoder reliability except during slip, approximate contact model, and dependency on covariance tuning.


## Implementation chapter
- include the noise parameter units and how those parameters can be taken from the IMU data sheet
  
  
  
  
## Possible questions:
- how the noise of wheel encoders modelled? is it used anywhere? should that not play a role somewhere?
- 