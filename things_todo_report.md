

## Background
- write about the inefk's propagate step, the state transition matrix and noise matrix and how they work

## Methodology
First draft is complete and sent for review

## Implementation chapter
- include the noise parameter units and how those parameters can be taken from the IMU data sheet

### table of contents for this chapter 

\chapter{Implementation}

\section{Software and System Setup}
\section{Orogen Task Structure}
    \subsection{Configuration Phase}
    \subsection{Runtime Update Loop}
    \subsection{Input and Output Ports}

\section{Filter Initialization}
    \subsection{Initial State}
    \subsection{Initial Covariance}
    \subsection{Noise Parameters}

\section{IMU Measurement Handling}z
    \subsection{IMU Frame Transformation}
    \subsection{Gyroscope Bias Estimation}
    \subsection{IMU Propagation}

\section{Robot Kinematics and Frame Handling}
    \subsection{Contact Frame Transformations}
    \subsection{Fixed Wheel Frames}
    \subsection{Foot-to-Wheel Transformations}
    \subsection{URDF-based Wheel Axis Extraction}

\section{Potential Contact Velocity Implementation}
    \subsection{Predicted PCV from Filter State}
    \subsection{Measured PCV from Wheel Encoders}
    \subsection{Residual Construction}

\section{Measurement Jacobian and Update Construction}
    \subsection{Observation Matrix}
    \subsection{Measurement Noise Matrix}
    \subsection{Stacked Measurement Update}

\section{Slip-aware Measurement Rejection}
    \subsection{Innovation Covariance}
    \subsection{Mahalanobis Distance Gate}
    \subsection{Accepted and Rejected Spokes}

\section{Contact Spoke Selection}
    \subsection{Lowest-Foot Selection}
    \subsection{Mahalanobis-Distance-based Selection}
    \subsection{Multiple Contact Handling}

\section{InEKF Library Modifications}
    \subsection{Added Contact Velocity Correction Interface}
    \subsection{Kalman Gain and Innovation Debug Outputs}

\section{CMAES Noise Parameter Optimization}
\section{Implementation Assumptions and Limitations}

  
## Possible questions:
- how the noise of wheel encoders modelled? is it used anywhere? should that not play a role somewhere?


