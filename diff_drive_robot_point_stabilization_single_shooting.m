clear,clc, close all;
addpath(pathdef)

import casadi.*

% Params
T = 0.2;        % sampling time
N = 100;          % prediction horizon
rob_diam = 0.3; % robot diameter

v_max = 0.6;
v_min = -v_max;         % m/s - linear vel limit

omega_max = pi/4;
omega_min = -omega_max; % rad/d - angular vel limit

% states
x = SX.sym('x');
y = SX.sym('y');
theta = SX.sym('theta');
states = [x; y; theta];
n_states = length(states);

% controls
v = SX.sym('v');         % linear vel
omega = SX.sym('omega'); % angular vel
controls = [v; omega];
n_controls = length(controls);

% state space sys
rhs = [
    v*cos(theta);
    v*sin(theta);
    omega
    ];
f = Function('f', {states, controls}, {rhs}, {'states', 'controls'}, {'rhs'}); % function to compute state derivative

% optimization problem setup
U = SX.sym('U', n_controls, N);   % decision variable
P = SX.sym('P', n_states*2);      % hold initial states and reference states
X = SX.sym('X', n_states, (N+1)); % rollouts of states during prediction horizon

% compute states trajectory symbolically
X(:,1) = P(1:3); % initial states
for k = 1:N
    st = X(:,k); % current state
    con = U(:,k); % current control
    stateDerivative = f(st, con);
    st_next = st + (T * stateDerivative); % Euler discretization to find next state
    X(:, k+1) = st_next;
end

ff = Function('ff', {U, P}, {X}, {'U', 'P'}, {'X'}); % function to compute the optimal state trajectory knowing the optimal controls

obj = 0;              % objective function
g = [];               % constraints vector

Q = diag([1,5,0.1]);   % states penalty
R = diag([0.5, 0.05]); % controls penalty

% compute cost function
for k = 1:N
    st = X(:, k);
    con = U(:, k);
    obj = obj + (st-P(4:end))'*Q*(st-P(4:end)) + con'*R*con;
end

% compute constraints
for k = 1:N+1         % box constraints due to map margins
    g = [g; X(1, k)]; % state x
    g = [g; X(2, k)]; % state y
end

% define solver
optVar = reshape(U, 2*N, 1); % this case optimization variables conveniently coincides with the controls

% define nonlinear programming problem
nlpProb = struct();
nlpProb.f = obj;
nlpProb.x = optVar;
nlpProb.g = g;
nlpProb.p = P;

% define solver options
opts.ipopt.max_iter = 100;
opts.ipopt.print_level = 0; % 0 -> 3
opts.ipopt.acceptable_tol = 1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

% define plugin options
opts.print_time = 0; % 0, 1

% define solver
solver = nlpsol('solver', 'ipopt', nlpProb, opts);

% define arguments
args = struct;

args.lbg = -2; % lower bound of states x and y (states constraints)
args.ubg = 2;  % upper bound of states x and y (states constraints)

args.lbx(1:2:2*N-1, 1) = v_min;   % lower bound of linear velocity (input constraints)
args.ubx(1:2:2*N-1, 1) = v_max;   % upper bound of linear velocity (input constraints)

args.lbx(2:2:2*N, 1) = omega_min; % lower bound of angular velocity (input constraints)
args.ubx(2:2:2*N, 1) = omega_max; % upper bound of angular velocity (input constraints)

% Simulation loop
% ------------------------------------------------------------------------------------
t0 = 0;
x0 = [0; 0; 0.0]; % initial condition
xs = [1.5; 1.5; 0.0]; % reference pose
xx(:, 1) = x0; % keep history of states during the whole simulation
u0 = zeros(N, 2); % two control inputs
t(1) = t0; % keep history of time
sim_time = 20; % maximum simulation time (sec)

% start MPC
mpcIter = 0;
xx1 = []; % to store predicted states
u_cl = []; % to store predicted controls we take (the first one in the all optimal controls)

main_loop = tic;
while((norm((x0 - xs), 2) > 1e-2) && ((mpcIter * T) < sim_time)) % terminate when the robot is within 1 cm of the destination or sim time is over
    args.p = [x0; xs]; % params containing init state and ref state
    args.x0 = reshape(u0', 2*N, 1); % initial value of the optimization variables
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx, 'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p);
    u = reshape(full(sol.x)', 2, N)';
    ff_value = ff(u', args.p); % compute optimal trajectory
    xx1(:, 1:3, mpcIter + 1) = full(ff_value)';
    u_cl = [u_cl; u(1, :)]; % store only the first control
    t(mpcIter + 1) = t0;
    [t0, x0, u0] = shift(T, t0, x0, u, f); % apply only the first control
    xx(:, mpcIter + 2) = x0;
    mpcIter = mpcIter + 1;
end
main_loop_time = toc(main_loop);
disp(main_loop_time/mpcIter);

Draw_MPC_point_stabilization_v1(t, xx, xx1, u_cl, xs, N, rob_diam);
