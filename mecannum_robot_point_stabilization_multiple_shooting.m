clear,clc, close all;
addpath(pathdef)

import casadi.*

% Params
T = 0.2;          % sampling time
N = 5;           % prediction horizon
rob_diam = 0.3;   % diameter of the robot
wheel_radius = 1; % wheel radius
Lx = 0.3;         % L in J Matrix (half robot x-axis length)
Ly = 0.3;         % l in J Matrix (half robot y-axis length)

v_max = 1;
v_min = -v_max;         % m/s - linear vel limit

Q = diag([100, 100, 2000]);   % states penalty
R = diag([1, 1, 1, 1]);     % controls penalty

% initial condition
limit_low = -2;
limit_high = 2;
x0 = [(limit_high-limit_low)*rand() + limit_low; ...
      (limit_high-limit_low)*rand() + limit_low; ...
      2*pi*rand()];   % initial pose

% target condition
xs = [(limit_high-limit_low)*rand() + limit_low; ...
      (limit_high-limit_low)*rand() + limit_low; ...
      2*pi*rand()];   % reference pose

% states
x = SX.sym('x');
y = SX.sym('y');
theta = SX.sym('theta');
states = [x; y; theta];
n_states = length(states);

% controls
V_a = SX.sym('V_a');
V_b = SX.sym('V_b');
V_c = SX.sym('V_c');
V_d = SX.sym('V_d');
controls = [V_a; V_b; V_c; V_d];
n_controls = length(controls);

% state space sys
rot_3d_z = [cos(theta), -sin(theta), 0; ...
            sin(theta),  cos(theta), 0; ...
                     0,           0, 1];

% Mecanum wheel transfer function which can be found here: 
% https://www.researchgate.net/publication/334319114_Model_Predictive_Control_for_a_Mecanum-wheeled_robot_in_Dynamical_Environments
J = (wheel_radius/4) .* [        1,         1,          1,         1; ...
                                -1,         1,          1,        -1; ...
                        -1/(Lx+Ly), 1/(Lx+Ly), -1/(Lx+Ly), 1/(Lx+Ly)];

X_dot = rot_3d_z * J * controls;
% maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
f = Function('f', {states, controls}, {X_dot}, {'states', 'controls'}, {'X_dot'}); % function to compute state derivative

% optimization problem setup
U = SX.sym('U', n_controls, N);   % decision variable (controls)
P = SX.sym('P', n_states*2);      % hold initial states and reference states
X = SX.sym('X', n_states, (N+1)); % rollouts of states during prediction horizon

obj = 0;              % objective function
g = [];               % constraints vector

st = X(:, 1); % initial states
g = [g; st - P(1:n_states)]; % initial condition constraint

for k = 1:N % calculate objective function and dynamics constraints
    st = X(:,k); % current state
    con = U(:,k); % current control
    obj = obj + (st-P(n_states+1:end))'*Q*(st-P(n_states+1:end)) + con'*R*con; % calculate objective function
    st_next = X(:, k+1); % next states to optimized
    st_next_RK4 = RK4(st, con, f, T);
    g = [g; st_next - st_next_RK4]; % dynamics constraints
end

% define solver
optVar = [reshape(X, n_states*(N+1), 1); ...
          reshape(U, n_controls*N, 1)]; % both X and U are optimization variable

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

% define constraints
args = struct;

args.lbg(1:n_states*(N+1)) = 0; % dynamics contraints (equality constraints)
args.ubg(1:n_states*(N+1)) = 0;

args.lbx(1:n_states:n_states*(N+1), 1) = -2;            % state x lower bound
args.ubx(1:n_states:n_states*(N+1), 1) = 2;             % state x upper bound
args.lbx(2:n_states:n_states*(N+1), 1) = -2;            % state y lower bound
args.ubx(2:n_states:n_states*(N+1), 1) = 2;             % state y upper bound
args.lbx(3:n_states:n_states*(N+1), 1) = -inf;          % state theta lower bound
args.ubx(3:n_states:n_states*(N+1), 1) = inf;           % state theta upper bound

args.lbx(n_states*(N+1)+1:(n_states*(N+1)+n_controls*N), 1) = v_min;     % lower bound for all v (all wheels)
args.ubx(n_states*(N+1)+1:(n_states*(N+1)+n_controls*N), 1) = v_max;     % upper bound for all v (all wheels)

% Simulation loop
% ------------------------------------------------------------------------------------
t0 = 0;

xx(:, 1) = x0;                 % keep history of states during the whole simulation
u0 = zeros(N, n_controls);     % two control inputs
X0 = repmat(x0, 1, N+1);       % init states variables along first prediction window
t(1) = t0;                     % keep history of time
sim_time = 20;                 % maximum simulation time (sec)

% start MPC
mpcIter = 0;
xx1 = []; % to store predicted states
u_cl = []; % to store predicted controls we take (the first one in the all optimal controls)

main_loop = tic;
while((norm((x0 - xs), 2) > rob_diam/10) && ((mpcIter * T) < sim_time)) % terminate when the robot is within 1 cm of the destination or sim time is over
    args.p = [x0; xs]; % params containing init state and ref state
    args.x0 = [reshape(X0', n_states*(N+1), 1); ...
               reshape(u0, n_controls*N, 1)]; % initial value of the optimization variables
    
    % solution consists of both optimal control (u_opt) and optimal trajectory (X_opt)
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx, ...
        'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p);

    % get only opt control from the solution
    u = reshape(full(sol.x(n_states*(N+1)+1:end))', n_controls, N)';

    % get only opt traj from the solution
    xx1(:, 1:n_states, mpcIter + 1) = reshape(full(sol.x(1:n_states*(N+1)))', n_states, N+1)';

    u_cl = [u_cl; u(1, :)]; % store only the first control

    t(mpcIter + 1) = t0;

    [t0, x0, u0] = shift(T, t0, x0, u + randn(1,n_controls)/50, f); % apply only the first control

    xx(:, mpcIter + 2) = x0;

    X0 = reshape(full(sol.x(1:n_states*(N+1)))', n_states, N+1)';
    X0 = [X0(2:end,:);X0(end,:)];

    mpcIter = mpcIter + 1;
end
main_loop_time = toc(main_loop);
disp(main_loop_time/mpcIter);

Draw_MPC_point_stabilization_v1(t, xx, xx1, u_cl, xs, N, rob_diam, false, "mecannum_robot_point_stabilization");


% TODO: Opti Stack