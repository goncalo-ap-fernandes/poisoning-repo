function results = N4SID()
%N4SID Evaluate transfer of the supplied poisoning profiles to a linear model.
% Requires MATLAB with System Identification Toolbox. No ANN training is run.
root = fileparts(mfilename('fullpath'));
output = fullfile(root, 'N4SID_Results');
mkdir(output);
mkdir(fullfile(output, 'Predictions'));
order = 2;
sampling_interval = 0.02;
window = 100;
physical_ids = [1, 10, 2, 3, 4, 5, 6];
clean_folder = fullfile(root, 'Datasets', 'Normal Datasets');
clean = cell(1, 7);
for index = 1:7
    clean{index} = readmatrix(fullfile(clean_folder, ...
        sprintf('DCMotor_%d.csv', physical_ids(index))));
end
test = readmatrix(fullfile(clean_folder, 'DCMotor_9.csv'));
indices = transition_indices(test(:,1), window);
assert(~isempty(indices), 'The test recording has no complete transition windows.');
results = table;

% One clean condition followed by 36 supplied poisoned conditions.
for tau = [0, 25, 50, 100]
    if tau == 0
        epsilons = 0;
        methods = "Normal";
    else
        epsilons = [0.05, 0.10, 0.15, 0.20];
        methods = ["Calculated", "Gaussian", "PRBS"];
    end
    for epsilon = epsilons
        for method = methods
            training = clean;
            profile_rms = 0;
            if method ~= "Normal"
                source_method = method;
                if method == "PRBS"
                    source_method = "Uniform"; % Original dataset naming.
                end
                folder = fullfile(root, 'Datasets', sprintf('top%d', tau), ...
                    sprintf('adversarialTop%d_%g', tau, epsilon), source_method);
                for index = 1:7
                    % Poisoned suffixes 1:7 correspond to physical_ids above.
                    training{index} = readmatrix(fullfile(folder, ...
                        sprintf('top%d_%s_%d.csv', tau, lower(source_method), index)));
                    difference = abs(training{index}(:,1:2) - clean{index}(:,1:2));
                    assert(max(difference(:)) < 1e-12, ...
                        'Poisoned and clean recordings do not match.');
                end
                % Recover the fixed rising/falling profiles from recording 1.
                reference = double(clean{1}(:,1) > 0.5);
                rising = find(diff([0; reference]) == 1, 1);
                falling = find(diff(reference) == -1, 1) + 1;
                delta = training{1}(:,3) - clean{1}(:,3);
                samples = [rising:rising+window-1, falling:falling+window-1];
                profile_rms = sqrt(mean(delta(samples).^2));
            end

            % Match the ANN float32 loader, then use MATLAB double precision.
            data = iddata(double(single(training{1}(:,3))), ...
                double(single(training{1}(:,2))), sampling_interval);
            for index = 2:7
                experiment = iddata(double(single(training{index}(:,3))), ...
                    double(single(training{index}(:,2))), sampling_interval);
                data = merge(data, experiment);
            end
            model = n4sid(data, order); % All identification options are defaults.
            prediction = free_simulation(model, test);
            target = double(single(test(:,3)));
            error = target - prediction;
            mse = mean(error.^2);
            transition_maxae = max(abs(error(indices)));
            row = table(method, tau, epsilon, profile_rms, mse, ...
                mean(error(indices).^2), max(abs(error)), transition_maxae, ...
                isstable(model), all(isfinite(prediction)), isfinite(mse) && mse <= 1, ...
                'VariableNames', {'method','tau','epsilon','profile_rms','mse', ...
                'transition_mse','maxae','transition_maxae','stable','finite','included'});
            results = [results; row]; %#ok<AGROW>
            name = sprintf('%s_top%d_%g.csv', lower(method), tau, epsilon);
            signals = table((0:numel(target)-1)' * sampling_interval, ...
                test(:,1), double(single(test(:,2))), target, prediction, ...
                'VariableNames', {'time','setpoint','input','target','prediction'});
            writetable(signals, fullfile(output, 'Predictions', name));
            writetable(results, fullfile(output, 'metrics.csv'));
            fprintf('%d/37 %s tau=%d epsilon=%g: MaxAE=%.9g\n', ...
                height(results), method, tau, epsilon, transition_maxae);
        end
    end
end
fprintf('Results: %s\n', output);
end

function indices = transition_indices(setpoint, width)
% Include an initially high reference; omit incomplete final windows.
reference = double(setpoint(:) > 0.5);
starts = find(diff(reference) ~= 0) + 1;
if reference(1) == 1
    starts = [1; starts];
end
starts = starts(starts + width - 1 <= numel(reference));
if isempty(starts)
    indices = zeros(0, 1);
    return;
end
% Concatenate windows as in the ANN metrics, retaining any overlap.
indices = reshape(starts + (0:width-1), [], 1);
end

function prediction = free_simulation(model, recording)
% Use only the first two outputs to initialize; never reset at transitions.
[A, B, C, D] = ssdata(model);
order = size(A, 1);
input = double(single(recording(:,2)));
target = double(single(recording(:,3)));
observability = zeros(order, order);
forced = zeros(order, 1);
state = zeros(order, 1);
for index = 1:order
    observability(index,:) = C * A^(index-1);
    forced(index) = C * state + D * input(index);
    state = A * state + B * input(index);
end
state = pinv(observability) * (target(1:order) - forced);
prediction = zeros(size(target));
for index = 1:numel(target)
    prediction(index) = C * state + D * input(index);
    state = A * state + B * input(index);
end
% Match the ANN convention: the measured prefix has zero reported error.
prediction(1:order) = target(1:order);
end
