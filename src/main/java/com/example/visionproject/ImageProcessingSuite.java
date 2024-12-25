package com.example.visionproject;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.*;
import javafx.scene.layout.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import java.io.File;
import javafx.scene.chart.*;
import java.util.Arrays;

public class ImageProcessingSuite extends Application {
    private ImageView imageView;
    private WritableImage currentImage;
    private LineChart<Number, Number> histogramChart;
    private PixelReader pixelReader;
    private PixelWriter pixelWriter;

    @Override
    public void start(Stage primaryStage) {
        VBox root = new VBox(10);
        setupGUI(root);
        Scene scene = new Scene(root, 1200, 800);
        primaryStage.setTitle("Image Processing Suite");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private void setupGUI(VBox root) {
        // Menu buttons
        HBox buttonBox = new HBox(10);
        Button loadButton = new Button("Load Image");
        Button grayButton = new Button("RGB to Gray");
        Button binaryButton = new Button("Gray to Binary");
        Button histogramButton = new Button("Show Histogram");
        Button equalizeButton = new Button("Histogram Equalization");
        Button erosionButton = new Button("Erosion");
        Button dilationButton = new Button("Dilation");
        Button sobelButton = new Button("Sobel Edge");
        Button cannyButton = new Button("Canny Edge");
        Button fourierButton = new Button("Fourier Transform");

        buttonBox.getChildren().addAll(
                loadButton, grayButton, binaryButton, histogramButton,
                equalizeButton, erosionButton, dilationButton,
                sobelButton, cannyButton, fourierButton
        );

        // Image display
        imageView = new ImageView();
        imageView.setFitWidth(800);
        imageView.setPreserveRatio(true);

        // Histogram setup
        NumberAxis xAxis = new NumberAxis();
        NumberAxis yAxis = new NumberAxis();
        histogramChart = new LineChart<>(xAxis, yAxis);
        histogramChart.setTitle("Image Histogram");
        histogramChart.setPrefHeight(300);

        // Button actions
        loadButton.setOnAction(e -> loadImage());
        grayButton.setOnAction(e -> convertToGrayscale());
        binaryButton.setOnAction(e -> convertToBinary());
        histogramButton.setOnAction(e -> showHistogram());
        equalizeButton.setOnAction(e -> equalizeHistogram());
        erosionButton.setOnAction(e -> applyErosion());
        dilationButton.setOnAction(e -> applyDilation());
        sobelButton.setOnAction(e -> applySobelEdge());
        cannyButton.setOnAction(e -> applyCannyEdge());
        fourierButton.setOnAction(e -> applyFourierTransform());

        root.getChildren().addAll(buttonBox, imageView, histogramChart);
    }

    private void loadImage() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg")
        );

        File file = fileChooser.showOpenDialog(null);
        if (file != null) {
            Image image = new Image(file.toURI().toString());
            currentImage = new WritableImage(
                    (int)image.getWidth(),
                    (int)image.getHeight()
            );
            pixelWriter = currentImage.getPixelWriter();
            pixelReader = image.getPixelReader();

            for (int y = 0; y < image.getHeight(); y++) {
                for (int x = 0; x < image.getWidth(); x++) {
                    pixelWriter.setArgb(x, y, pixelReader.getArgb(x, y));
                }
            }
            imageView.setImage(currentImage);
        }
    }

    private void convertToGrayscale() {
        if (currentImage == null) return;

        WritableImage grayImage = new WritableImage(
                (int)currentImage.getWidth(),
                (int)currentImage.getHeight()
        );
        PixelWriter writer = grayImage.getPixelWriter();

        for (int y = 0; y < currentImage.getHeight(); y++) {
            for (int x = 0; x < currentImage.getWidth(); x++) {
                int argb = pixelReader.getArgb(x, y);
                int r = (argb >> 16) & 0xff;
                int g = (argb >> 8) & 0xff;
                int b = argb & 0xff;
                int gray = (int)(0.299 * r + 0.587 * g + 0.114 * b);
                int grayArgb = (0xff << 24) | (gray << 16) | (gray << 8) | gray;
                writer.setArgb(x, y, grayArgb);
            }
        }
        currentImage = grayImage;
        imageView.setImage(currentImage);
    }

    private void convertToBinary() {
        if (currentImage == null) return;

        WritableImage binaryImage = new WritableImage(
                (int)currentImage.getWidth(),
                (int)currentImage.getHeight()
        );
        PixelWriter writer = binaryImage.getPixelWriter();

        // Otsu's thresholding
        int threshold = calculateOtsuThreshold();

        for (int y = 0; y < currentImage.getHeight(); y++) {
            for (int x = 0; x < currentImage.getWidth(); x++) {
                int argb = pixelReader.getArgb(x, y);
                int gray = (argb >> 16) & 0xff; // Assuming image is grayscale
                int binary = gray > threshold ? 255 : 0;
                int binaryArgb = (0xff << 24) | (binary << 16) | (binary << 8) | binary;
                writer.setArgb(x, y, binaryArgb);
            }
        }
        currentImage = binaryImage;
        imageView.setImage(currentImage);
    }

    private int calculateOtsuThreshold() {
        int[] histogram = new int[256];

        // Calculate histogram
        for (int y = 0; y < currentImage.getHeight(); y++) {
            for (int x = 0; x < currentImage.getWidth(); x++) {
                int gray = (pixelReader.getArgb(x, y) >> 16) & 0xff;
                histogram[gray]++;
            }
        }

        // Otsu's method
        int total = (int)(currentImage.getWidth() * currentImage.getHeight());
        float sum = 0;
        for (int i = 0; i < 256; i++) sum += i * histogram[i];

        float sumB = 0;
        int wB = 0;
        int wF = 0;
        float varMax = 0;
        int threshold = 0;

        for (int i = 0; i < 256; i++) {
            wB += histogram[i];
            if (wB == 0) continue;
            wF = total - wB;
            if (wF == 0) break;

            sumB += i * histogram[i];
            float mB = sumB / wB;
            float mF = (sum - sumB) / wF;

            float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);

            if (varBetween > varMax) {
                varMax = varBetween;
                threshold = i;
            }
        }

        return threshold;
    }

    private void showHistogram() {
        if (currentImage == null) return;

        // Calculate histogram
        int[] histogram = new int[256];
        for (int y = 0; y < currentImage.getHeight(); y++) {
            for (int x = 0; x < currentImage.getWidth(); x++) {
                int gray = (pixelReader.getArgb(x, y) >> 16) & 0xff;
                histogram[gray]++;
            }
        }

        // Update chart
        XYChart.Series<Number, Number> series = new XYChart.Series<>();
        for (int i = 0; i < 256; i++) {
            series.getData().add(new XYChart.Data<>(i, histogram[i]));
        }

        histogramChart.getData().clear();
        histogramChart.getData().add(series);
    }

    private void equalizeHistogram() {
        if (currentImage == null) return;

        int width = (int)currentImage.getWidth();
        int height = (int)currentImage.getHeight();
        int totalPixels = width * height;

        // Calculate histogram
        int[] histogram = new int[256];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = (pixelReader.getArgb(x, y) >> 16) & 0xff;
                histogram[gray]++;
            }
        }

        // Calculate cumulative histogram
        int[] cdf = new int[256];
        cdf[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cdf[i] = cdf[i-1] + histogram[i];
        }

        // Create lookup table
        int[] lookup = new int[256];
        for (int i = 0; i < 256; i++) {
            lookup[i] = Math.round(((float)cdf[i] / totalPixels) * 255);
        }

        // Apply equalization
        WritableImage equalizedImage = new WritableImage(width, height);
        PixelWriter writer = equalizedImage.getPixelWriter();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = (pixelReader.getArgb(x, y) >> 16) & 0xff;
                int newGray = lookup[gray];
                int newArgb = (0xff << 24) | (newGray << 16) | (newGray << 8) | newGray;
                writer.setArgb(x, y, newArgb);
            }
        }

        currentImage = equalizedImage;
        imageView.setImage(currentImage);
    }

    private void applyErosion() {
        if (currentImage == null) return;

        int width = (int)currentImage.getWidth();
        int height = (int)currentImage.getHeight();
        WritableImage erodedImage = new WritableImage(width, height);
        PixelWriter writer = erodedImage.getPixelWriter();

        // 3x3 structuring element
        for (int y = 1; y < height-1; y++) {
            for (int x = 1; x < width-1; x++) {
                int minVal = 255;

                // Find minimum in 3x3 neighborhood
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int pixel = (pixelReader.getArgb(x+dx, y+dy) >> 16) & 0xff;
                        minVal = Math.min(minVal, pixel);
                    }
                }

                int newArgb = (0xff << 24) | (minVal << 16) | (minVal << 8) | minVal;
                writer.setArgb(x, y, newArgb);
            }
        }

        currentImage = erodedImage;
        imageView.setImage(currentImage);
    }

    private void applyDilation() {
        if (currentImage == null) return;

        int width = (int)currentImage.getWidth();
        int height = (int)currentImage.getHeight();
        WritableImage dilatedImage = new WritableImage(width, height);
        PixelWriter writer = dilatedImage.getPixelWriter();

        // 3x3 structuring element
        for (int y = 1; y < height-1; y++) {
            for (int x = 1; x < width-1; x++) {
                int maxVal = 0;

                // Find maximum in 3x3 neighborhood
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int pixel = (pixelReader.getArgb(x+dx, y+dy) >> 16) & 0xff;
                        maxVal = Math.max(maxVal, pixel);
                    }
                }

                int newArgb = (0xff << 24) | (maxVal << 16) | (maxVal << 8) | maxVal;
                writer.setArgb(x, y, newArgb);
            }
        }

        currentImage = dilatedImage;
        imageView.setImage(currentImage);
    }

    private void applySobelEdge() {
        if (currentImage == null) return;

        int width = (int)currentImage.getWidth();
        int height = (int)currentImage.getHeight();
        WritableImage edgeImage = new WritableImage(width, height);
        PixelWriter writer = edgeImage.getPixelWriter();

        // Sobel operators
        int[][] sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int[][] sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        for (int y = 1; y < height-1; y++) {
            for (int x = 1; x < width-1; x++) {
                int gx = 0, gy = 0;

                // Apply operators
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        int pixel = (pixelReader.getArgb(x+j, y+i) >> 16) & 0xff;
                        gx += pixel * sobelX[i+1][j+1];
                        gy += pixel * sobelY[i+1][j+1];
                    }
                }

                // Calculate magnitude
                int magnitude = (int)Math.min(255, Math.sqrt(gx*gx + gy*gy));
                int newArgb = (0xff << 24) | (magnitude << 16) | (magnitude << 8) | magnitude;
                writer.setArgb(x, y, newArgb);
            }
        }

        currentImage = edgeImage;
        imageView.setImage(currentImage);
    }
    private void applyCannyEdge() {
        if (currentImage == null) return;

        int width = (int)currentImage.getWidth();
        int height = (int)currentImage.getHeight();

        // 1. Gaussian blur
        double[][] gaussian = new double[][]{
                {1/16.0, 2/16.0, 1/16.0},
                {2/16.0, 4/16.0, 2/16.0},
                {1/16.0, 2/16.0, 1/16.0}
        };
        WritableImage blurredImage = applyFilter(gaussian);

        // 2. Sobel Edge Detection
        int[][] sobelX = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int[][] sobelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        double[][] magnitude = new double[width][height];
        double[][] direction = new double[width][height];
        double maxMagnitude = 0;

        PixelReader blurredReader = blurredImage.getPixelReader();

        for (int y = 1; y < height-1; y++) {
            for (int x = 1; x < width-1; x++) {
                double gx = 0, gy = 0;

                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        int pixel = (blurredReader.getArgb(x+j, y+i) >> 16) & 0xff;
                        gx += pixel * sobelX[i+1][j+1];
                        gy += pixel * sobelY[i+1][j+1];
                    }
                }

                magnitude[x][y] = Math.sqrt(gx*gx + gy*gy);
                maxMagnitude = Math.max(maxMagnitude, magnitude[x][y]);
                direction[x][y] = Math.atan2(gy, gx);
            }
        }

        // 3. Non-maximum suppression
        WritableImage suppressedImage = new WritableImage(width, height);
        PixelWriter suppressedWriter = suppressedImage.getPixelWriter();

        for (int y = 1; y < height-1; y++) {
            for (int x = 1; x < width-1; x++) {
                double angle = direction[x][y] * 180 / Math.PI;
                if (angle < 0) angle += 180;

                double mag = magnitude[x][y];
                double mag1 = 0, mag2 = 0;

                if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) {
                    mag1 = magnitude[x][y-1];
                    mag2 = magnitude[x][y+1];
                }
                else if (22.5 <= angle && angle < 67.5) {
                    mag1 = magnitude[x+1][y-1];
                    mag2 = magnitude[x-1][y+1];
                }
                else if (67.5 <= angle && angle < 112.5) {
                    mag1 = magnitude[x-1][y];
                    mag2 = magnitude[x+1][y];
                }
                else if (112.5 <= angle && angle < 157.5) {
                    mag1 = magnitude[x-1][y-1];
                    mag2 = magnitude[x+1][y+1];
                }

                int gray = (mag >= mag1 && mag >= mag2) ?
                        (int)(mag * 255 / maxMagnitude) : 0;

                int argb = (0xff << 24) | (gray << 16) | (gray << 8) | gray;
                suppressedWriter.setArgb(x, y, argb);
            }
        }

        // 4. Double threshold and edge tracking
        double highThreshold = maxMagnitude * 0.15;
        double lowThreshold = highThreshold * 0.4;

        WritableImage edgeImage = new WritableImage(width, height);
        PixelWriter edgeWriter = edgeImage.getPixelWriter();

        for (int y = 1; y < height-1; y++) {
            for (int x = 1; x < width-1; x++) {
                int pixel = (suppressedImage.getPixelReader().getArgb(x, y) >> 16) & 0xff;
                double mag = (pixel / 255.0) * maxMagnitude;

                int gray;
                if (mag >= highThreshold) {
                    gray = 255;  // Strong edge
                } else if (mag >= lowThreshold) {
                    // Check if connected to strong edge
                    boolean connected = false;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0) continue;
                            int neighborPixel = (suppressedImage.getPixelReader().getArgb(x+dx, y+dy) >> 16) & 0xff;
                            double neighborMag = (neighborPixel / 255.0) * maxMagnitude;
                            if (neighborMag >= highThreshold) {
                                connected = true;
                                break;
                            }
                        }
                        if (connected) break;
                    }
                    gray = connected ? 255 : 0;
                } else {
                    gray = 0;
                }

                int argb = (0xff << 24) | (gray << 16) | (gray << 8) | gray;
                edgeWriter.setArgb(x, y, argb);
            }
        }

        currentImage = edgeImage;
        imageView.setImage(currentImage);
    }

    private void applyFourierTransform() {
        if (currentImage == null) return;

        int width = (int)currentImage.getWidth();
        int height = (int)currentImage.getHeight();

        // Convert image to complex numbers
        Complex[][] complex = new Complex[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = (pixelReader.getArgb(x, y) >> 16) & 0xff;
                complex[y][x] = new Complex(pixel, 0);
            }
        }

        // Apply 2D FFT
        complex = fft2D(complex);

        // Convert to magnitude spectrum
        WritableImage spectrumImage = new WritableImage(width, height);
        PixelWriter writer = spectrumImage.getPixelWriter();

        double maxMagnitude = 0;
        double[][] magnitudes = new double[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                magnitudes[y][x] = Math.log(1 + complex[y][x].abs());
                maxMagnitude = Math.max(maxMagnitude, magnitudes[y][x]);
            }
        }

        // Normalize and display
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int magnitude = (int)(magnitudes[y][x] * 255 / maxMagnitude);
                int argb = (0xff << 24) | (magnitude << 16) | (magnitude << 8) | magnitude;
                writer.setArgb(x, y, argb);
            }
        }

        currentImage = spectrumImage;
        imageView.setImage(currentImage);
    }

    // Helper class for complex numbers
    private static class Complex {
        private final double re;
        private final double im;

        public Complex(double real, double imag) {
            re = real;
            im = imag;
        }

        public Complex add(Complex b) {
            return new Complex(re + b.re, im + b.im);
        }

        public Complex subtract(Complex b) {
            return new Complex(re - b.re, im - b.im);
        }

        public Complex multiply(Complex b) {
            return new Complex(re * b.re - im * b.im, re * b.im + im * b.re);
        }

        public double abs() {
            return Math.sqrt(re * re + im * im);
        }
    }

    // 2D FFT implementation
    private Complex[][] fft2D(Complex[][] input) {
        int height = input.length;
        int width = input[0].length;
        Complex[][] result = new Complex[height][width];

        // FFT on rows
        for (int y = 0; y < height; y++) {
            result[y] = fft1D(input[y]);
        }

        // FFT on columns
        for (int x = 0; x < width; x++) {
            Complex[] column = new Complex[height];
            for (int y = 0; y < height; y++) {
                column[y] = result[y][x];
            }
            column = fft1D(column);
            for (int y = 0; y < height; y++) {
                result[y][x] = column[y];
            }
        }

        return result;
    }

    // 1D FFT implementation (Cooley-Tukey algorithm)
    private Complex[] fft1D(Complex[] x) {
        int n = x.length;

        if (n == 1) return new Complex[] { x[0] };

        Complex[] even = new Complex[n/2];
        Complex[] odd = new Complex[n/2];
        for (int k = 0; k < n/2; k++) {
            even[k] = x[2*k];
            odd[k] = x[2*k + 1];
        }

        Complex[] q = fft1D(even);
        Complex[] r = fft1D(odd);
        Complex[] y = new Complex[n];

        for (int k = 0; k < n/2; k++) {
            double kth = -2 * k * Math.PI / n;
            Complex wk = new Complex(Math.cos(kth), Math.sin(kth));
            y[k] = q[k].add(wk.multiply(r[k]));
            y[k + n/2] = q[k].subtract(wk.multiply(r[k]));
        }

        return y;
    }

    private WritableImage applyFilter(double[][] kernel) {
        int width = (int)currentImage.getWidth();
        int height = (int)currentImage.getHeight();
        WritableImage filteredImage = new WritableImage(width, height);
        PixelWriter writer = filteredImage.getPixelWriter();

        int kernelSize = kernel.length;
        int padding = kernelSize / 2;

        for (int y = padding; y < height-padding; y++) {
            for (int x = padding; x < width-padding; x++) {
                double sum = 0;

                for (int ky = 0; ky < kernelSize; ky++) {
                    for (int kx = 0; kx < kernelSize; kx++) {
                        int pixel = (pixelReader.getArgb(x + kx - padding, y + ky - padding) >> 16) & 0xff;
                        sum += pixel * kernel[ky][kx];
                    }
                }

                int value = (int)Math.min(255, Math.max(0, sum));
                int argb = (0xff << 24) | (value << 16) | (value << 8) | value;
                writer.setArgb(x, y, argb);
            }
        }

        return filteredImage;
    }

    public static void main(String[] args) {
        launch(args);
    }
}