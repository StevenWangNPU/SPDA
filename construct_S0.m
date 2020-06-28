function  S0 =  construct_S0( idx, h, ni)
% initial a graph in which the weight of nearest
% neighborhood is one.
S0 = zeros(ni,ni);
for j = 1:ni
    for k = 1:h
        S0(j, idx(j,k+1)) = 1/h;
    end
end
