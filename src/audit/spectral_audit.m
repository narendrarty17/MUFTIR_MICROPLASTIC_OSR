%% ========================================================================
%% PROJECT: Replication of µFTIR Microplastic Classification
%% PURPOSE: Data Ingestion, Mapping, and Visualization (Step 00)
%% ========================================================================

clear; clc; close all;

% Define common file names used across scripts
h5File = 'PET_fibers_2.hdf5'; 

%% ------------------------------------------------------------------------
%% SCRIPT 1: SINGLE SAMPLE MAPPING (Testing Phase)
%% Use this to verify the alignment of one sample before batch processing.
%% ------------------------------------------------------------------------
fprintf('--- Running Script 1: Single Sample Mapping ---\n');

% Define the exact dataset names identified via h5disp
waveName = '/wavenumbers'; 
sampleName = '/99'; % Change this to any numeric ID (e.g., '98', '97')

try
    wavenumbers = h5read(h5File, waveName);
    intensities = h5read(h5File, sampleName);
    
    % Create Table (Aligning 882 points)
    testTable = table(wavenumbers, intensities, ...
        'VariableNames', {'Wavenumber_cm_1', 'Intensity'});
    
    writetable(testTable, 'Single_Sample_Check.xlsx');
    fprintf('Success: Single sample mapped to "Single_Sample_Check.xlsx"\n\n');
catch ME
    fprintf('Error in Script 1: %s\n\n', ME.message);
end


%% ------------------------------------------------------------------------
%% SCRIPT 2: BATCH PROCESSING (Audit Phase)
%% Automatically finds all numeric datasets and maps them to one Excel file.
%% ------------------------------------------------------------------------
fprintf('--- Running Script 2: Batch Processing ---\n');

info = h5info(h5File);
allDatasets = {info.Datasets.Name};

% Master X-axis (Wavenumbers)
wavenumbers = h5read(h5File, '/wavenumbers');

% Find all numeric datasets (e.g., '1', '2', ..., '99')
isNumeric = ~isnan(str2double(allDatasets));
sampleNames = allDatasets(isNumeric);

% Initialize table with Wavenumbers
batchTable = table(wavenumbers, 'VariableNames', {'Wavenumber_cm_1'});

for i = 1:length(sampleNames)
    sName = sampleNames{i};
    try
        data = h5read(h5File, ['/' sName]);
        if length(data) == length(wavenumbers)
            % Table column names cannot start with numbers; adding 'S_' prefix
            batchTable.(['S_' sName]) = data;
        end
    catch
        % Skips corrupted or non-matching datasets
    end
end

writetable(batchTable, 'Full_Mapped_Dataset.xlsx');
fprintf('Success: %d samples consolidated into "Full_Mapped_Dataset.xlsx"\n\n', length(sampleNames));


%% ------------------------------------------------------------------------
%% SCRIPT 3: SPECTRAL VISUALIZATION (Verification Phase)
%% Plots 10 samples to check for noise and alignment for the 1D-CNN.
%% ------------------------------------------------------------------------
fprintf('--- Running Script 3: Visualization ---\n');

numToPlot = min(10, length(sampleNames));
figure('Name', 'µFTIR Spectral Audit');
hold on;

for i = 1:numToPlot
    plot(wavenumbers, h5read(h5File, ['/' sampleNames{i}]), ...
        'DisplayName', ['Sample ', sampleNames{i}]);
end

% Professional Formatting for Research
title(['µFTIR Spectra: First ', num2str(numToPlot), ' Samples']);
xlabel('Wavenumber (cm^{-1})');
ylabel('Intensity (Absorbance)');
legend('Location', 'northeastoutside');
grid on;

% Standard FTIR convention: Invert the X-axis
set(gca, 'XDir', 'reverse'); 

% Save plot as high-res PNG
saveas(gcf, 'Spectral_Audit_Plot.png');
hold off;

fprintf('Success: Spectral diagram saved as "Spectral_Audit_Plot.png"\n');
fprintf('========================================================================\n');