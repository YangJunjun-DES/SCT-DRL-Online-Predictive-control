% M1

Com = {};
for i = 1:2
    for j=1:2
        for k=1:2
            for m=1:7
                for n=1:2
                    Com{end+1}=[i,j,k,m,n];
                 end
            end
        end
    end
end
 
% Event = {};
% Event_u = {};
% for j = 1:192
%     [Events_available,Events_available_u] = Events(state,state_space,P1,P2,P3,B1SUP,B2SUP);
%     Event{end+1} = Events_available;
%     Event_u{end+1} = Events_available_u;
% end















