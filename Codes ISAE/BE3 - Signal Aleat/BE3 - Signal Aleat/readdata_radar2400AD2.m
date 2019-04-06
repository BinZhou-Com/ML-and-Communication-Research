function [Fc,B,Tr,L,M,s_mix_1,s_mix_2] = readdata_radar2400AD2(filename,option)
%READDATA_RADAR2400AD2 Read Ancortek data Radar kit 2400AD2.
% [FC,B,TR,L,M,S_MIX_1,S_MIX_2] = READDATA_RADAR2400AD2(FILENAME) returns
% carrier frequency FC, bandwidth B for FMCW (resp., step frequency or zero
% for FSK and CW), sweep time TR, range gate numbers L, and mixed signal
% S_MIX_1 and S_MIX_2 on both received channels. 
% All parameters are in IS units. 
% Can read data recorded either by Ancortek or Malab-Gui.
%
% [...] = READDATA_RADAR2400AD2(FILENAME,OPTION) writes also in the command
% window the scenario parameters if print option is 'on'. If not specified,
% the default print option is 'off'.
%
% Note: fix the TODO.

% 2017-02-13  | stephanie.bidon@isae-supaero.fr

%% Retrieve file extension
[~,~,ext] = fileparts(filename);

%% Read data
if strcmp(ext,'.dat') % ancortek-gui
    
    fileID = fopen(filename,'r');
    dataArray = textscan(fileID,'%f');
    fclose(fileID);
    radarData = dataArray{1};
    clearvars fileID dataArray;
    %--
    Fc = radarData(1);              % (Hz) center frequency
    B = radarData(4);               % (Hz) FMCW-bandwidth or FSK-frequency step or zero-CW-bandwidth
    Tr = radarData(2)*1e-3;         % (s) sweep time
    L = radarData(3);               % (-) number of time samples per sweep
    %--
    M = ((length(radarData)-4)/(2*L));      % (-) pulses number
    %--
    s_iq = conj(radarData(5:end));        % raw data in I+jQ format
    %--
    s_mix_1 = s_iq(1:2:end);         % data of channel 1
    s_mix_2 = s_iq(2:2:end);         % data of channel 2
    clear s_iq
    
elseif strcmp(ext,'.mat') % matlab-gui
    
    raw = load(filename);
    %--
    Fc = raw.CENTERFREQUENCY;
    Tr = raw.SWEEPTIME * 1e-3;
    L = raw.samplenumberpersweep;
    B = raw.BANDWIDTH;
    s_mix_1 = conj(raw.DATA1); % data of channel 1
    s_mix_2 = conj(raw.DATA2); % data of channel 2
    M = length(s_mix_1)/L;
    
else
    error('Unknown filename extension.')
end

%% Display scenario parameters
if nargin==1
    option = 'off';
end
if strcmp(option,'on')
    fprintf('%s\n',repmat('-',76,1))
    if strcmp(ext,'.mat')
        fprintf('Date: %s\n',raw.DATE);
        fprintf('Waveform: %s\n',raw.WAVEFORM);
    end
    fprintf('Carrier frequency:\t%.2f (GHz)\n',Fc*1e-9);
    fprintf('Bandwidwth:\t\t%.2f (GHz)\n',B*1e-9);%TODO: ajust print if not FMCW
    fprintf('PRI:\t\t\t%.2f (ms)\n',Tr*1e3);
    fprintf('Samples per PRI:\t%d (-)\n',L);
    fprintf('Number of pulses:\t%d (-)\n',M);
    %--
    c = 3e8;
    va = c/(2*Fc)/Tr;           % (m/s) ambiguous velocity
    Ra = c*Tr/2;                % (m) ambiguous range
    %--
    fprintf('Maximum range:\t\t%.2f (m)\n',c/2/B*L/2);
    fprintf('Ambiguous velocity:\t%.2f (m/s)\n',va);
    fprintf('Ambiguous range:\t%.2f (m)\n',Ra);
    fprintf('%s\n',repmat('-',76,1))
end

