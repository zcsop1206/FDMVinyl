% generate_openvinyl_plots.m
% Script to generate high-contrast visualizations for the OpenVinyl portfolio
% Saves the plots to the specified output directory

clear; close all; clc;

%% Configuration
output_dir = 'C:\Users\Adit Bhargava\Documents\EngineeringPortfolio\src\assets\openvinyl';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fs = 44100;         % Sample rate
t = 0:1/fs:0.1;     % 100ms time vector
f0 = 1000;          % 1kHz test tone
x = 0.8 * sin(2*pi*f0*t); % Input signal

% High contrast plot settings
bg_color = [0.05 0.05 0.05];
fg_color = [0.9 0.9 0.9];
grid_color = [0.2 0.2 0.2];
line_color = [0.2 0.8 1.0]; % Cyan-ish for main plots

%% Helper function to configure plot aesthetics
function config_plot(fig, ax, bg_color, fg_color, grid_color)
    set(fig, 'Color', bg_color, 'InvertHardcopy', 'off');
    set(ax, 'Color', bg_color, 'XColor', fg_color, 'YColor', fg_color, ...
        'GridColor', grid_color, 'GridAlpha', 0.8, 'MinorGridColor', grid_color, ...
        'MinorGridAlpha', 0.5, 'LineWidth', 1.2);
    grid on;
    box on;
end

%% Plot A: Naive Quantization (The Problem)
num_levels = 16; % 4-bit = 16 levels
mapped = (x + 1.0) * 0.5 * (num_levels - 1);
x_quant_naive = round(mapped);
x_quant_naive = (x_quant_naive / (num_levels - 1)) * 2.0 - 1.0;

figure('Position', [100, 100, 800, 400], 'Name', 'Naive Quantization', 'Color', bg_color);
ax1 = axes;

% Calculate FFT
N = length(x_quant_naive);
n_win = 0:(N-1);
blackman_win = 0.42 - 0.5 * cos(2*pi*n_win/(N-1)) + 0.08 * cos(4*pi*n_win/(N-1));
X_k = fft(x_quant_naive .* blackman_win);
mag_X = 20*log10(abs(X_k(1:N/2+1)) / (N/2) + eps);
f_axis = (0:N/2)*fs/N;

plot(f_axis, mag_X, 'Color', [0.9 0.3 0.3], 'LineWidth', 1.5);
title('FFT Spectrum: Naively Quantized 4-Bit Signal', 'Color', fg_color, 'FontSize', 14);
xlabel('Frequency (Hz)', 'Color', fg_color, 'FontSize', 12);
ylabel('Magnitude (dB)', 'Color', fg_color, 'FontSize', 12);
ylim([-100, 0]);
xlim([0, 20000]);

config_plot(gcf, ax1, bg_color, fg_color, grid_color);
exportgraphics(gcf, fullfile(output_dir, 'plot_a_naive_quant.png'), 'Resolution', 300);

%% Plot B: Butterworth Anti-Aliasing Filter (The Filter)
cutoff_fg = 8000;
filter_order = 4;

figure('Position', [100, 100, 800, 400], 'Name', 'Butterworth Filter', 'Color', bg_color);
ax2 = axes;

% Calculate magnitude response analytically to avoid Signal Processing Toolbox
f_ideal = linspace(0, fs/2, 2048);
mag_h = 10 * log10(1 ./ (1 + (f_ideal / cutoff_fg).^(2 * filter_order)) + eps);

w = f_ideal; % Map for the plot function below

plot(w, mag_h, 'Color', [0.2 0.8 1.0], 'LineWidth', 2.0);
title('Frequency Response: 4th-Order Butterworth Filter (8kHz Cutoff)', 'Color', fg_color, 'FontSize', 14);
xlabel('Frequency (Hz)', 'Color', fg_color, 'FontSize', 12);
ylabel('Magnitude (dB)', 'Color', fg_color, 'FontSize', 12);
ylim([-80, 5]);
xlim([0, 20000]);
xline(8000, '--', 'Color', [0.8 0.8 0.2], 'LineWidth', 1.5); % Cutoff freq

config_plot(gcf, ax2, bg_color, fg_color, grid_color);
exportgraphics(gcf, fullfile(output_dir, 'plot_b_filter.png'), 'Resolution', 300);

%% Plot C: Dithering + 2nd Order Lipshitz Noise Shaping (The Solution)
% Apply TPDF (Triangular Probability Density Function) Dither
dither = rand(1, length(x)) - rand(1, length(x));
dither_amp = 0.5 / (num_levels - 1); % 1 LSB p-p
x_dithered = x + dither * dither_amp;

x_out = zeros(size(x_dithered));
e1 = 0; e2 = 0;
c1 = 1.5; c2 = -0.5;

for i = 1:length(x_dithered)
    val = (x_dithered(i) + 1.0) * 0.5 * (num_levels - 1);
    val_shaped = val + c1 * e1 + c2 * e2;
    q = round(val_shaped);
    q = max(0, min(num_levels-1, q));
    e2 = e1;
    e1 = val_shaped - q;
    x_out(i) = (q / (num_levels - 1)) * 2.0 - 1.0;
end

figure('Position', [100, 100, 800, 400], 'Name', 'Noise Shaping', 'Color', bg_color);
ax3 = axes;

X_k_shaped = fft(x_out .* blackman_win);
mag_X_shaped = 20*log10(abs(X_k_shaped(1:N/2+1)) / (N/2) + eps);

plot(f_axis, mag_X_shaped, 'Color', [0.3 0.9 0.4], 'LineWidth', 1.5);
title('FFT Spectrum: 2nd-Order Lipshitz Noise Shaping + TPDF Dither', 'Color', fg_color, 'FontSize', 14);
xlabel('Frequency (Hz)', 'Color', fg_color, 'FontSize', 12);
ylabel('Magnitude (dB)', 'Color', fg_color, 'FontSize', 12);
ylim([-100, 0]);
xlim([0, 20000]);

config_plot(gcf, ax3, bg_color, fg_color, grid_color);
exportgraphics(gcf, fullfile(output_dir, 'plot_c_noise_shaping.png'), 'Resolution', 300);

disp('Plots generated and saved successfully.');
