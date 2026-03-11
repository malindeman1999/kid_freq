function Z = resonance(f,fr,Q,Rat,a_amp,a_arg,tau,phi) 
#Rat= Q/abs(Qcom) phi=arg(Q/Qcom)=-arg(Qcom) phi is not the phase of Qcom, but the phase of 1/Qcom
#according to Khalil eqn 11 and 12
#assume function of form Gao's theses, Khalil paper
a_parm=a_amp*exp(i*a_arg);
Z=a_parm.*exp(i.*2.*pi*f.*tau).*(1-Rat.*exp(i.*phi)./(1.+i.*2*Q.*((f.-fr)./fr)));
endfunction
