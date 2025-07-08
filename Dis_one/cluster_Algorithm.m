

num_transtions = 0;
OBSER = 1;
OBSER1 = 1;
Method = 2;


%rollout
RO_nodes  = 2;
RO_traces = 30;
RO_depth = 3;
RO_gamma = gamma;
n_actions = 4;


while(~isempty(OBSER))
    state = OBSER(1);
    isDone = 0;
    %the available events by SCT
    obs = State_space(state, :);
    [Event_set,Enb_P] = AllowedEvnts(obs,P1,P2,P3,R_B1SUP,R_B2SUP);

    if Method == 1 %Only optimal action  6 states, 7 transitions
        optimal_action = choose_optimal_action(state,Q_table); 
        
    else  %rollout  72 states, 182 transitions
        [pattern_value, optimal_action] = rollout_test(Q_table, state, RO_nodes, ...
            RO_traces, RO_depth, RO_gamma, n_actions, R, State_space);
    end

    %Compute the corresponding patrern
    if optimal_action ~= 4
        Controllable_event = E_c(optimal_action);
        pattern = setdiff(Event_set, Controllable_event);
    else
        pattern = Event_set;
    end
    
    %execute all events in the generated pattern
    for event_idx = 1 : length(pattern)
        event = pattern(event_idx);
        obs1_ = StepFunction(P1,P2,P3,R_B1SUP,R_B2SUP,obs,event);
        [~,state1_] = ismember(obs1_,State_space,"rows");
        [Event_set_,~] = AllowedEvnts(obs1_,P1,P2,P3,R_B1SUP,R_B2SUP);
        if isempty(Event_set_)
            fprintf("A deadlock occurs!\n");
            isDone = 1;
            break
        else
            if (~ismember(state1_,OBSER1)) 
                OBSER1(end+1) = state1_;
                OBSER(end+1) = state1_;
                 
            end
         fprintf('%d->%d[label=%d];\n',state,state1_,event);
         num_transtions = num_transtions + 1; 
        end

    end
    OBSER(1) = [];
    %break if a deadlock occurs
    if isDone == 1
        break
    end


    % move to the next state
    policy = pattern(unidrnd(length(pattern)));
    obs_ = StepFunction(P1, P2, P3, R_B1SUP, R_B2SUP, obs, policy);
    obs = obs_;
    [~,state_] = ismember(obs_,State_space,"rows");
    state = state_;
end
