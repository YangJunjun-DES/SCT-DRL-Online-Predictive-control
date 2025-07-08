

num_transtions = 0;
OBSER = 1;
OBSER1 = 1;
Method = 2;

while(~isempty(OBSER))
    state = OBSER(1);
    isDone = 0;
    %the available events by SCT
    obs = State_space(state, :);
    [Event_set,Enb_P] = AllowedEvnts(obs,P1,P2,P3,R_B1SUP,R_B2SUP);
    
    optimal_action = choose_optimal_action(state,Q_table);
    Event_subset = B{optimal_action};
    pattern = intersect(Event_set, Event_subset);
    pattern = union(pattern, E_u);
    pattern = intersect(pattern, Enb_P);
    
    
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
