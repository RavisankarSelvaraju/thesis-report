

## Background
- write about the inefk's propagate step, the state transition matrix and noise matrix and how they work

## Methodology


\section{Potential Contact Velocity Measurement}
This is the core methodology section.

\subsection{Predicted PCV}
Explain how PCV is predicted from the filter state.
Define the velocity of a potential contact point using body velocity, angular velocity, and the contact point position.

\subsection{Measured PCV}
Explain how PCV is derived from wheel encoder measurements.
Define the relation between wheel angular velocity and spoke/contact-point velocity.

\subsection{Residual / Innovation}
Explain that the difference between measured PCV and predicted PCV forms the residual used in the filter update.

\section{Measurement Model and Jacobian}
Present the measurement function.
Present or reference the analytical Jacobian.
Mention that the analytical Jacobian is numerically verified.

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
- 