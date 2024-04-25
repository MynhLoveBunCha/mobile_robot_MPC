clear,clc, close all;
addpath(pathdef)

import casadi.*

% Params
T = 0.2;        % sampling time
N = 25;          % prediction horizon
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
X_dot = [
    v*cos(theta);
    v*sin(theta);
    omega
    ];
f = Function('f', {states, controls}, {X_dot}, {'states', 'controls'}, {'X_dot'}); % function to compute state derivative

% optimization problem setup
U = SX.sym('U', n_controls, N);   % decision variable (controls)
P = SX.sym('P', n_states*2);      % hold initial states and reference states
X = SX.sym('X', n_states, (N+1)); % rollouts of states during prediction horizon

obj = 0;              % objective function
g = [];               % constraints vector

Q = diag([5, 5, 0.1]);   % states penalty
R = diag([0.5, 0.01]);     % controls penalty

st = X(:, 1); % initial states
g = [g; st - P(1:3)]; % initial condition constraint

for k = 1:N % calculate objective function and dynamics constraints
    st = X(:,k); % current state
    con = U(:,k); % current control
    obj = obj + (st-P(4:end))'*Q*(st-P(4:end)) + con'*R*con; % calculate objective function
    st_next = X(:, k+1); % next states to optimized
    st_next_RK4 = RK4(st, con, f, T);
    % stateDerivative = f(st, con);
    % st_next_euler = st + (T * stateDerivative); % predicted next states
    g = [g; st_next - st_next_RK4]; % dynamics constraints
end

% define solver
optVar = [reshape(X, 3*(N+1), 1); reshape(U, 2*N, 1)]; % both X and U are optimization variable

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

args.lbg(1:3*(N+1)) = 0; % dynamics contraints (equality constraints)
args.ubg(1:3*(N+1)) = 0;

args.lbx(1:3:3*(N+1), 1) = -2;            % state x lower bound
args.ubx(1:3:3*(N+1), 1) = 2;             % state x upper bound
args.lbx(2:3:3*(N+1), 1) = -2;            % state y lower bound
args.ubx(2:3:3*(N+1), 1) = 2;             % state y upper bound
args.lbx(3:3:3*(N+1), 1) = -inf;          % state theta lower bound
args.ubx(3:3:3*(N+1), 1) = inf;           % state theta upper bound

args.lbx(3*(N+1)+1:2:3*(N+1)+2*N, 1) = v_min;     % input v lower bound
args.ubx(3*(N+1)+1:2:3*(N+1)+2*N, 1) = v_max;     % input v upper bound
args.lbx(3*(N+1)+2:2:3*(N+1)+2*N, 1) = omega_min; % input omega lower bound
args.ubx(3*(N+1)+2:2:3*(N+1)+2*N, 1) = omega_max; % input omega upper bound

% Simulation loop
% ------------------------------------------------------------------------------------
t0 = 0;
x0 = [0; 0; 0.0];        % initial condition

limit_low = -2;
limit_high = 2;
xs_x = (limit_high-limit_low)*rand() + limit_low;
xs_y = (limit_high-limit_low)*rand() + limit_low;
xs_theta = 2*pi*rand();
xs = [xs_x; xs_y; xs_theta]    % reference pose
xx(:, 1) = x0;                 % keep history of states during the whole simulation
u0 = zeros(N, 2);              % two control inputs
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
    args.x0 = [reshape(X0', 3*(N+1), 1); reshape(u0, 2*N, 1)]; % initial value of the optimization variables
    
    % solution consists of both optimal control (u_opt) and optimal trajectory (X_opt)
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx, 'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p);

    % get only opt control from the solution
    u = reshape(full(sol.x(3*(N+1)+1:end))', 2, N)';

    % get only opt traj from the solution
    xx1(:, 1:3, mpcIter + 1) = reshape(full(sol.x(1:3*(N+1)))', 3, N+1)';

    u_cl = [u_cl; u(1, :)]; % store only the first control

    t(mpcIter + 1) = t0;

    [t0, x0, u0] = shift(T, t0, x0, u + randn(1,2)/10, f); % apply only the first control

    xx(:, mpcIter + 2) = x0;

    X0 = reshape(full(sol.x(1:3*(N+1)))', 3, N+1)';
    X0 = [X0(2:end,:);X0(end,:)];

    mpcIter = mpcIter + 1;
end
main_loop_time = toc(main_loop);
disp(main_loop_time/mpcIter);

Draw_MPC_point_stabilization_v1(t, xx, xx1, u_cl, xs, N, rob_diam, false, "diff_drive_point_stabilization");

% TODO: change to mecanum wheel
% TODO: cartPend