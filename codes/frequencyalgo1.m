function f = frequencyalgo1(sn,Q)
N = length(sn);
Sn = fft(sn);
magSn = abs(Sn); 
Yn = magSn.^2;
[peaks,locs]=findpeaks(Yn);
[~,locs_sorted] = sort(peaks,'descend');
plocs = locs - ones(size(locs));
mcap = plocs(locs_sorted(1));
b = plocs(locs_sorted(2));
% M = [a,b];
% mcap = min(M);
delta = zeros(1,Q+1);
xp05 =0;
xn05 =0;

for i = 1:1:Q
    for k = 1:N
        xp05 = xp05 + sn(1,k).*exp((-1.i).*(2*pi).*(k-1).*((mcap+delta(1,i)+0.5)/N));
        xn05 = xn05 + sn(1,k).*exp((-1.i).*(2*pi).*(k-1).*((mcap+delta(1,i)-0.5)/N));
    end
   p = (xp05 + xn05)/(xp05-xn05);
   h = (1/2)*(real(p));
   delta(1,i+1) = delta(1,i) + h;
    
end

f = ((mcap+delta(Q+1))/N);

end

