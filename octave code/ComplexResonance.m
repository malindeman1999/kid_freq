function S21 = ComplexResonance(f,fr,Q,Qcom,a,tau) 

#Q is the resonator total Q
#Qcom is the complex valued coupling Q.  (Qe~ from Khalil)

#This function serves as a wrapper for resonance
#it takes normal parameters and translates them in to the parameters I assigned 
#resonance function which the the core of my fitting routines
#
#Rat= Q/abs(Qcom) phi=arg(Q/Qcom)=-arg(Qcom) phi is not the phase of Qcom, but the phase of 1/Qcom
#according to Khalil eqn 11 and 12
#assume function of form Gao's theses, Khalil paper

a_amp=abs(a);
a_arg=arg(a);
Rat=abs(Q/Qcom);
phi=arg(Q/Qcom);



a_parm=a_amp*exp(i*a_arg);
S21=resonance(f,fr,Q,Rat,a_amp,a_arg,tau,phi) ;
endfunction
