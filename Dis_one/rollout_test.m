function [pattern_value, pattern_index] = rollout_test(Q_table, State, RO_nodes, RO_traces, RO_depth, RO_gamma, n_actions, R, State_space)

    
load("P1.mat");   load("P2.mat"); load("P3.mat"); %Components
load("R_B1SUP.mat");  load("R_B2SUP.mat"); %Supervisors
E_u = [2,4,6,8];
E_c = [1,3,5];
    
    % 获取通过 Q 表获得的指导动作选择的前 N 个动作
    pattern_value = Q_table(State, :);
    [~, sorted_indices] = sort(-pattern_value); % 降序排列
    top_N_indices = sorted_indices(1:RO_nodes);
    pattern_value = zeros(1, n_actions);
    pattern_count = zeros(1, n_actions);
    RO_nodes_num = RO_nodes;
    
    % 执行 Monte Carlo rollout，运行 RO_traces 次，深度为 RO_depth
    for i = 1:RO_traces
        if ~isempty(top_N_indices)
            try
                act_idx_ori = top_N_indices(mod(i-1, RO_nodes_num) + 1);
                act_idx = act_idx_ori;
                pattern_count(act_idx_ori) = pattern_count(act_idx_ori) + 1;
                Q_eval_ro = 0;
                skip_outer_loop = false; % 控制外循环的标志
            catch
                disp('An error occurred in the main function');
            end
        else
            % 如果 top_N_indices 为空，表示所有顶部节点死锁
            break;
        end
        state = State;
        for RO_i = 1:RO_depth 
            obs = State_space(state, :);
            %compute the corresponding pattern
            [Event_set, ~] = AllowedEvnts(obs,P1,P2,P3,R_B1SUP, R_B2SUP);
            if act_idx ~= 4
                Controllable_event = E_c(act_idx);
                pattern = setdiff(Event_set, Controllable_event); 
            else
                pattern = Event_set;
            end
            
            if ~isempty(pattern) && all(ismember(pattern, Event_set))
                %randomly choose an event form the selected available events
                Action = pattern(unidrnd(length(pattern)));
                %execute the selected action and observe the next state
                RO_S_ = StepFunction(P1,P2,P3,R_B1SUP,R_B2SUP,obs,Action);
                
                %is_Done_test
                is_Done_test = 0;
                for j = 1:length(pattern)
                    event = pattern(j);
                    Next_obs = StepFunction(P1,P2,P3,R_B1SUP,R_B2SUP,obs,event);  %One possible observation
                    [Event_set_,~] = AllowedEvnts(Next_obs,P1,P2,P3,R_B1SUP, R_B2SUP);
                    if isempty(Event_set_)
                        is_Done_test = 1;
                        break
                    end    
                end
            else
                is_Done_test = 1;
            end
            
            %get step reward    
            R_t = length(pattern) ./ n_actions;
            if is_Done_test
                R_t = -100;
            else
                for k = 1 : length(pattern) 
                    i_action = pattern(k);
                    i_reward = R(i_action);
                    R_t = R_t + i_reward;
                end
            end
            
            % 如果检测到阻塞状态，则从候选列表中删除它并分配负的 V 值
            if is_Done_test
                rollout_method = 1; % 1 分配负值，2 切断路径
                if rollout_method == 1
                    pattern_value(act_idx_ori) = pattern_value(act_idx_ori) + (-100 * (RO_gamma ^ (RO_i-1)));
                else
                    pattern_value(act_idx_ori) = -inf;
                    top_N_indices(top_N_indices == act_idx_ori) = [];
                    RO_nodes_num = RO_nodes_num - 1;
                end
                skip_outer_loop = true; % 设定标志为 true
                break;
            end            
            Q_eval_ro = Q_eval_ro + (RO_gamma ^ (RO_i-1)) * R_t;  %1-step lookahead
            [~,State_] = ismember(RO_S_, State_space,"rows"); 
            if RO_i + 1 == RO_depth
                pattern_value_ = Q_table(State_, :);
                break; %This trace is terminated
            else
                act_idx = find(Q_table(State_, :) == max(Q_table(State_, :)), 1);%rollout
                state = State_;
            end
        end
        
        if skip_outer_loop
            continue; % 跳过外循环的下一次迭代
        end
    
        Q_eval_ro = Q_eval_ro + (RO_gamma ^ RO_depth) * max(pattern_value_);
        pattern_value(act_idx_ori) = pattern_value(act_idx_ori) + Q_eval_ro;
    end
    
    pattern_value(top_N_indices) = pattern_value(top_N_indices) ./ pattern_count(top_N_indices);
    
    Obs = State_space(State, :);
    % 根据 pattern_value 选择 pattern_index
    if ~isempty(top_N_indices)
        [~, pattern_length] = Q_value_eval(Q_table, Obs, R, E_c, E_u);        
        [~, ~, pattern_index] = pattern_index_select(Q_table, pattern_value, Obs, pattern_length,State_space);
    else
        pattern_index = [];
    end
end








