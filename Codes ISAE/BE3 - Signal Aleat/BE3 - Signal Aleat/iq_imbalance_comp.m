function [y,phi] = iq_imbalance_comp(x)
%IQ_IMBALANCE_COMP IQ imbalance compensation.
% Y = IQ_IMBALANCE_COMP(X) returns the complex signal Y that corresponds
% after IQ imbalance compensation.
%
% [Y,PHI] = IQ_IMBALANCE_COMP(X) returns also the angle PHI of the IQ
% imbalance in radians.

% 2017-07-29|stephanie.bidon@isae-supaero.fr
% References
% http://ancortek.com/wp-content/uploads/2015/06/Collecting-Data-using-Ancorteks-SDR-C-API.mp4

%%
I = real(x);
Q = imag(x);
%--
m_I = mean(I);
m_Q = mean(Q);
v_I = mean((I-m_I).^2);
v_Q = mean((Q-m_Q).^2);
v_IQ = mean((I-m_I).*(Q-m_Q));
D_bar = v_IQ/v_I;
C_bar = sqrt(v_Q/v_I-D_bar^2);
d_ampImb = sqrt(C_bar^2+D_bar^2)-1;
phi = atan(D_bar/C_bar);
I = I - m_I;
Q = ( (Q-m_Q)/(1+d_ampImb) -I*sin(phi))/cos(phi);

%%
y = I+1i*Q;

