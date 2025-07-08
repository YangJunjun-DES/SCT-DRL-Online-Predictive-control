%
load("P1.mat");   load("P2.mat"); load("P3.mat"); %Components
load("R_B1SUP.mat");  load("R_B2SUP.mat"); %Supervisors

Event_list = [1,2,1,2,1,2,1,2,3,4,3,4,1,2,1,2];
Init_s = [1,1,1,1,1];
obs = Init_s;
for i = 1:length(Event_list) 
    event = Event_list(i);
    Next_obs = StepFunction(P1,P2,P3,R_B1SUP,R_B2SUP,obs,event);
    obs = Next_obs;
end
[Event_set,~] = AllowedEvnts(obs,P1,P2,P3,R_B1SUP,R_B2SUP);
%No event is available after executing the Event_list.

