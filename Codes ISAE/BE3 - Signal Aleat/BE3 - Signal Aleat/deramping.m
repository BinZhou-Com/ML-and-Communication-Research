%DERAMPING Deramping technique.
% Fill in the "todo" and visually find the target in the range-time
% intensity plot obtained.
%
% See also readdata_radar2400AD2, iq_imbalance_comp.

% 2017-08-31 |stephanie.bidon@isae-supaero.fr
% 20??-??-?? |student@isae-supaero.fr ?? 
%todo: write your names

clearvars;clc;close all

fIQcomp = 1;    % flag for IQ-compensation

%% Read parameters and signal
datadir = '..';
%todo: write the name of the data directory
filename = fullfile(datadir,'2017-07-27-17-07-46.mat');
%--
[Fc,B,Tr,L,M,s_mix_1,s_mix_2] = readdata_radar2400AD2(filename,'on');

%% Useful parameters

% Physics
c = 3e8
;                    % (m/s) speed of light

% Radar
dR =  c/(2*B);              % (m) range resolution
va = c/(2*Fc)/Tr;           % (m/s) ambiguous velocity
Ra = c*Tr/2;                % (m) ambiguous range

%% Processing parameters

% Range profile
L_zp = 8*L;                         % (-) number of zeropadded range-gate
rg_zp = ((0:1/L_zp:1-1/L_zp)-.5)*L; % (-) zeropadded range-gate
range_zp = rg_zp*dR;              % (m) range

%% IQ-imbalance compensation
if fIQcomp
    s_mix_1 = iq_imbalance_comp(s_mix_1);
    s_mix_2 = iq_imbalance_comp(s_mix_2);
end

%% IQ data
h_fig1 = figure(1);
set(h_fig1,'Name','IQ data','visible','on')
%--
subplot(2,1,1)
plot(real(s_mix_1));hold on
plot(imag(s_mix_1));
xlabel('sample (-)')
legend('$I_1$','$Q_1$')
title('RX 1')
xlim([0 1500])
%--
subplot(2,1,2)
plot(real(s_mix_2));hold on
plot(imag(s_mix_2));
xlabel('sample (-)')
legend('$I_2$','$Q_2$')
title('RX 2')
xlim([0 1500])
%--
pause(.1)

%% Reshape the mixed signal in "datacube"
s_mix_1 = reshape(s_mix_1,L,M);   % L time-samples * M sweeps
s_mix_2 = reshape(s_mix_2,L,M);

%% Range transform
data_1 = ...;
data_2 = ...;
%todo: apply an IFFT to obtain the range profile (on L_zp points) on both
%RX channels

%% Display range-time intensity plot

clim = [0 60];
%todo: should be adapted if you have not normalized the IFFT operation

figure;
%--
subplot(2,1,1)
imagesc(1:M,range_zp,20*log10(ifftshift(abs(data_1),1)),clim);
axis xy
colormap(flipud(hot))
hc = colorbar;set(get(hc,'title'),'string','(dB)');
xlabel('sweep index (-)')
ylabel('range (m)')
ylim([0 range_zp(end)]);
title('RX 1')
grid off
%--
subplot(2,1,2)
imagesc(1:M,range_zp,20*log10(ifftshift(abs(data_2),1)),clim);
axis xy
colormap(flipud(hot))
hc = colorbar;set(get(hc,'title'),'string','(dB)');
xlabel('sweep index (-)')
ylabel('range (m)');
title('RX 2');
ylim([0 range_zp(end)]);
grid off
%--
suptitle('Range-time intensity plot')

%% Display a range cut at sweep m0
m0 = 11674;
y_lim = [0 65];
%todo: should be adapted if you have not normalized the IFFT operation

figure;
%--
subplot(2,1,1)
plot(range_zp,20*log10(ifftshift(abs(data_2(:,m0+1)),1)));
xlabel('range (m)')
xlim([0 range_zp(end)]);
ylabel('Intensity (dB)');
title('RX 1')
grid off
ylim(y_lim)
%--
subplot(2,1,2)
plot(range_zp,20*log10(ifftshift(abs(data_2(:,m0+1)),1)));
xlabel('range (m)')
xlim([0 range_zp(end)]);
ylabel('Intensity (dB)');
title('RX 2');
grid off
ylim(y_lim)
%--
suptitle(sprintf('Range profile - sweep %d',m0));

