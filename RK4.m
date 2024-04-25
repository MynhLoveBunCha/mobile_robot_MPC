function statesNext = RK4(statesCur, controlCur, dynFunc, Ts)
%RK4 4th-order Runge Kutta discretization 
%   calculate next state from current state and current control
k1 = dynFunc(statesCur, controlCur);
k2 = dynFunc(statesCur + Ts/2*k1, controlCur);
k3 = dynFunc(statesCur + Ts/2*k2, controlCur);
k4 = dynFunc(statesCur + Ts*k3, controlCur);
statesNext = statesCur +Ts/6*(k1 +2*k2 +2*k3 +k4);
end