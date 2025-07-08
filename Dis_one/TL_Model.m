clear;clc;
load("P1.mat");   load("P2.mat"); load("P3.mat"); %Components
load("R_B1SUP.mat");  load("R_B2SUP.mat"); %Supervisors

State_space = zeros(1,5);
State_reach = zeros(1,5);
initial_s = [1,1,1,1,1];
State_space(1,:) = initial_s;
State_reach(1,:) = initial_s;

while(~isempty(State_reach))

    State = State_reach(1,:);
    X1 = State(1); 
    X2 = State(2); 
    X3 = State(3);
    X4 = State(4); 
    X5 = State(5);
    
    
    Enable_P1 = AvailableEvents(X1, P1);
    Enable_P2 = AvailableEvents(X2, P2);
    Enable_P3 = AvailableEvents(X3, P3);
    Enable_P = union(union(Enable_P1,Enable_P2),Enable_P3);     
    
    Enable_B1SUP = AvailableEvents(X4, R_B1SUP);
    Enable_B2SUP = AvailableEvents(X5, R_B2SUP);
    
    Enable = intersect(Enable_B1SUP,Enable_B2SUP);   % Available events allowed by modular supervisors at current global state
    Enable_P_S = intersect(Enable,Enable_P);       
    
    
    for i = 1: length(Enable_P_S)
        event = Enable_P_S(i);
        State_ = StepFunction(P1,P2,P3,R_B1SUP,R_B2SUP,State,event);  
        if ~ismember(State_,State_space,"rows")
            State_space = [State_space;State_];
            State_reach = [State_reach;State_];
        end
    end
    State_reach(1,:) = [];
end
    








