# Importing The Important Libraries
# ---------------------------------
from PyQt5 import uic                                       # Importing User Interface Compiler module from PyQt5 for loading .ui files.
from PyQt5.QtWidgets import QMainWindow, QApplication       # Importing essential widgets, file dialog, and dialog window classes from PyQt5.
from PyQt5.QtWidgets import QFileDialog, QDialog            # Importing essential widgets, file dialog, and dialog window classes from PyQt5.
from PyQt5.QtGui import QIcon                               # Importing QIcon class for handling icons in the PyQt5 framework.
from PyQt5.QtCore import QTimer                             # Importing QTimer for creating and managing application timers in PyQt5.
import pyqtgraph as pg                                      # Importing pyqtgraph for interactive graphing, plotting, and data visualization.
from pyqtgraph import ScatterPlotItem                       # Importing ScatterPlotItem for creating scatter plot elements in pyqtgraph.
from scipy.interpolate import interp1d                      # Importing interpolation and filtering functions from SciPy for data processing.
from scipy.signal import butter, lfilter                    # Importing Filtering Signals in Digital Signal Processing
import wfdb                                                 # Importing wfdb for reading and writing files in PhysioBank format.
import numpy as np                                          # Importing NumPy for numerical operations on arrays and matrices.
from math import floor                                      # Importing floor function from math module for mathematical floor operation.
import pandas as pd







# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------






# MAIN CLASS FOR THE SAMPLING SIGNAL STUDIO
# -----------------------------------------
class SignalStudio(QMainWindow):
    def __init__(self):
        super(SignalStudio, self).__init__()
        # Set up UI from a .ui file, set window title and icon, and display the window.
        uic.loadUi("./SignalStudio.ui", self)
        self.setWindowTitle("Signal Studio")
        icon = QIcon("./Files/App_Icon.jpg")
        self.setWindowIcon(icon)
        self.show()

        # Establish connections between UI elements and their respective handlers.
        self.BrowseButton.clicked.connect(self.open_dialog_box)
        self.PlotSlider.valueChanged.connect(self.slider_moved)
        self.ClearButton.clicked.connect(self.clear_handler)
        self.AddSinusoidalButton.clicked.connect(self.add_sinusoidal)
        # Use lambda to pass the slider value as an argument to the noise addition function.
        self.SNRSlider.valueChanged.connect(lambda value: self.add_noise(value))
        self.SamplingSlider.valueChanged.connect(self.adjust_sampled_points)
        self.RemoveSinusoidalButton.clicked.connect(self.remove_sinusoidal)
        self.NyquistCheckbox.stateChanged.connect(self.nyquist)
        self.connect_sampling_widgets()

        # Initialize a timer for sampling control and set up data structures for signal manipulation.
        self.sampling_timer = QTimer(self)
        self.sampling_timer.setInterval(100)
        self.sampling_timer.setSingleShot(True)
        self.sampling_timer.timeout.connect(self.update_sampled_points)
        self.current_slider_value = 0
        self.sinusoidals = []  # List to hold sinusoidal functions added by the user.
        self.fmax = 0          # Variable to track the maximum frequency in the sinusoidals list.
        self.update_sampling_slider_max()



# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    #  Open a file dialog to allow the user to select a .dat file and process it accordingly.
    # ---------------------------------------------------------------------------------------
    def open_dialog_box(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # Ensure the dialog is read-only to prevent file modifications.
        # Launch the dialog to select a data file and store the selected path.
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "DAT Files (*.dat);;CSV Files (*.csv);;All Files (*)", options=options)
        if path:  # If a file was selected, determine how to process the signal based on existing plots.
            if self.Channel1Widget.plotItem.curves:
                self.mix_signal(path)  # If a curve is already plotted, mix the new signal.
            else:
                self.plot_signal(path)  # If no curves exist, simply plot the new signal.



# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Load and process a signal from a .dat file, updating class attributes with signal data.
    # ---------------------------------------------------------------------------------------
    def load_dat_signal(self, path):
        # Read the first channel of the signal from the .dat file, excluding the extension.
        record = wfdb.rdrecord(path[:-4], channels=[0])
        # Read the header information to obtain metadata.
        header = list(wfdb.rdheader(path[:-4]).comments)
        # Store sampling rate and concatenate select header information for display.
        self.sampling_rate = record.fs
        self.record_header = header[0] + ', ' + header[1] + ', ' + header[2]
        # Flatten the signal array and calculate its duration and corresponding time vector.
        self.signal = record.p_signal.flatten()
        self.duration = len(self.signal) / self.sampling_rate
        self.time = np.linspace(0, self.duration, len(self.signal))
        # Append the loaded signal to the list of sinusoidals for further operations.
        self.sinusoidals.append(self.signal)
        # Update the maximum frequency attribute if necessary.
        if self.sampling_rate / 2 > self.fmax:
            self.fmax = self.sampling_rate / 2
        # Return the signal and its time vector for further use.
        return self.signal, self.time
    

    def load_csv_signal(self, path):
        data = pd.read_csv(path)
        self.signal = np.array(pd.to_numeric(data.iloc[:, 1].values, downcast="float"))
        self.time = np.array(pd.to_numeric(data.iloc[:, 0].values, downcast="float"))
        self.sampling_rate = 1 / (self.time[1] - self.time[0])
        self.record_header = str(self.sampling_rate)
        self.fmax = self.sampling_rate / 2
        self.sinusoidals.append(self.signal)
        return self.signal, self.time

    


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
    
    
    
    # Configure plot appearance and settings for the given channel with optional background grid for the first channel.
    # -----------------------------------------------------------------------------------------------------------------
    def plot_settings(self, channel, time, signal, title, FIRST_CHANNEL=False):
        # Set plot limits to match the signal's time and amplitude range with some padding.
        channel.setLimits(xMin=0, xMax=time[-1], yMin=np.min(signal) - 0.2, yMax=np.max(signal) + 0.2)
        channel.setTitle(title)  # Assign the provided title to the plot.
        # Apply additional settings for the primary channel plot, if indicated.
        if FIRST_CHANNEL:
            channel.setBackground(None)  # Set the background to default.
            channel.showGrid(x=True, y=True)  # Display the grid for better visualization.



# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Plotting The Signal
    # -------------------
    def plot_signal(self, path):
        if path.endswith('.dat'):
            self.signal, self.time = self.load_dat_signal(path)
        elif path.endswith('.csv'):
            self.signal, self.time = self.load_csv_signal(path)

        
        # Clear Widget 1 and 2 
        self.Channel1Widget.clear()
        self.Channel2Widget.clear()
        
        # Plotting The Signal Update on Widget ONE
        self.Channel1Widget.plot(self.time, self.signal, pen=pg.mkPen(color='g'))
        self.plot_settings(self.Channel1Widget, self.time, self.signal, self.record_header, True)
        sampled_points_x = self.time
        sampled_points_y = self.signal
        print(len(self.time))
        self.Channel1Widget.plot(sampled_points_x, sampled_points_y, pen=None, symbol='x', symbolPen='w',
                                 symbolBrush=0.2, symbolSize=1)
        
        # Plotting The Signal Update on Widget TWO
        self.Channel2Widget.plot(sampled_points_x, sampled_points_y, pen=None, symbol='x', symbolPen='w',
                                 symbolBrush=0.2, symbolSize=1)
        self.Channel2Widget.setLimits(xMin=0, xMax=self.time[-1], yMin=np.min(self.signal) - 0.2, yMax=np.max(self.signal) + 0.2)
        
        # Link Between Both Widgets
        self.Channel1Widget.setXLink(self.Channel2Widget)
        self.Channel1Widget.setYLink(self.Channel2Widget)
        self.mixed_signal_with_noise = self.signal

        # Rename The Widget Title Name
        self.Channel1Widget.setTitle("Original Signal")

        # Ploting The Sampling Points in The ORIGINAL Signal
        sampled_scatter = ScatterPlotItem()
        sampled_scatter.setData(self.time, self.signal, symbol='o', brush=(255, 0, 0), size=5)
        self.Channel1Widget.addItem(sampled_scatter)



# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # # Mix a new signal with the existing signal on the first channel and update the plot accordingly.
    # -------------------------------------------------------------------------------------------------
    def mix_signal(self, path):
        # Retrieve existing signal data from the first channel's plot.
        plot_data_item = self.Channel1Widget.plotItem.curves[0]
        existing_time = plot_data_item.xData
        existing_signal = plot_data_item.yData

        # Load a new signal from the specified file path.
        record = wfdb.rdrecord(path[:-4], channels=[0])
        new_signal = record.p_signal.flatten()

        # Mix the existing signal with the new signal.
        mixed_signal = existing_signal + new_signal
        mixed_time = existing_time  # Assume the time vectors are identical.

        # Clear the channel plot and plot the mixed signal.
        self.Channel1Widget.clear()
        self.Channel1Widget.plot(mixed_time, mixed_signal, pen=pg.mkPen(color='w'), name='Mixed Signal')
        # Adjust plot limits for the mixed signal and add a legend.
        self.Channel1Widget.setLimits(xMin=0, xMax=mixed_time[-1], yMin=np.min(mixed_signal) - 0.2, yMax=np.max(mixed_signal) + 0.2)
        self.Channel1Widget.addLegend()

        # Store the mixed signal's curve and plot the individual signals with different colors but hidden by default.
        mixed_curve = self.Channel1Widget.getPlotItem().listDataItems()[-1]
        signal1_curve = self.Channel1Widget.plot(existing_time, existing_signal, pen=pg.mkPen(color='g', width=2), name='Signal 1')
        signal1_curve.setVisible(False)
        signal2_curve = self.Channel1Widget.plot(existing_time, new_signal, pen=pg.mkPen(color='r', width=2), name='Signal 2')
        signal2_curve.setVisible(False)

        # Add scatter points for the mixed signal on the plot.
        sampled_scatter = ScatterPlotItem()
        sampled_scatter.setData(mixed_time, mixed_signal, symbol='o', brush=(255, 0, 0), size=5)
        self.Channel1Widget.addItem(sampled_scatter)




# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



        
        
        # Update The Visibility of The Added Signal From Local Files (More Than ONE Signal)
        # ---------------------------------------------------------------------------------
        # Toggle visibility of signal plots in the legend based on user interaction with checkboxes.
        def update_visibility(item):
            # Check the label of the item and update visibility of the corresponding curve.
            if item.text() == "Signal 1":
                signal1_curve.setVisible(item.checkState() == pg.QtCore.Qt.Checked)
            elif item.text() == "Signal 2":
                signal2_curve.setVisible(item.checkState() == pg.QtCore.Qt.Checked)
            elif item.text() == "Mixed Signal":
                mixed_curve.setVisible(item.checkState() == pg.QtCore.Qt.Checked)

        # Retrieve the legend from the plot item of the channel.
        legend = self.Channel1Widget.getPlotItem().legend
        # Connect the state change of each checkbox in the legend to the visibility update function.
        for item, _ in legend.items:
            if isinstance(item, pg.LegendItem):
                for checkbox, label in item.items:
                    checkbox.sigStateChanged.connect(update_visibility)




# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Slider Value Movement Steps
    # ---------------------------
    # Adjust the visible range on the plot as the slider is moved to zoom in or out on the x-axis.
    def slider_moved(self, value):
        # Calculate the visible range as a percentage of the total range (20 units here).
        visible_range = value / 100 * 20
        # Update the x-axis range for both channels to the new visible range without padding.
        self.Channel1Widget.setXRange(self.current_slider_value, self.current_slider_value + visible_range, padding=0)
        self.Channel2Widget.setXRange(self.current_slider_value, self.current_slider_value + visible_range, padding=0)




# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Clear Button Handling & Actions
    # -------------------------------
    # Reset all channels, sliders, and internal states to their initial conditions.
    def clear_handler(self):
        # Clear the plots on all channel widgets.
        self.Channel1Widget.clear()
        self.Channel2Widget.clear()
        self.Channel3Widget.clear()
        # Reset the sliders to their minimum value, effectively 'zooming out' and removing noise.
        self.SamplingSlider.setValue(0)
        self.PlotSlider.setValue(0)
        self.SNRSlider.setValue(0)
        # Clear any stored mixed signal with noise and the list of sinusoidal signals.
        self.mixed_signal_with_noise = None
        self.sinusoidals = []
        # Reset the maximum frequency to zero and clear the sinusoidal combobox options.
        self.fmax = 0
        self.SinusoidalCombobox.clear()




# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Add Sinusoidal Signal on the Original Signal + Handling The Events
    # ------------------------------------------------------------------
    # Add a sinusoidal signal to the existing plot, or initiate a new plot if none exists.
    def add_sinusoidal(self):
        # Check if there are existing plots and set flag accordingly.
        if self.Channel1Widget.plotItem.curves:
            plot_data_item = self.Channel1Widget.plotItem.curves[0]
            existing_time = plot_data_item.xData
            existing_signal = plot_data_item.yData
            flag = 1  # Indicates that there is an existing signal
        else:
            existing_time = []
            existing_signal = []
            flag = 0  # Indicates a new signal will be initiated

        # Open a dialog to preview and add the sinusoidal signal.
        dialog = SignalPreviewDialog(self, existing_time, existing_signal, flag, self.sinusoidals)
        dialog.exec_()  # Execute the dialog as a modal window.

        # After dialog completion, update the mixed signal on the plot.
        self.update_mixed_signal()

        # Update the maximum value of the SamplingSlider to be 4 times the max frequency
        self.SamplingSlider.setMaximum(4 * floor(self.fmax))

# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
        


    # Update The Mixed Signal
    # -----------------------
    # Update and display the composed signal, adding all individual sinusoidal signals together.
    def update_mixed_signal(self):
        # Initialize the signal array, either as zeros matching the first sinusoidal array or as a default size.
        self.signal = np.zeros_like(self.sinusoidals[0]) if self.sinusoidals else np.zeros(1000)
        # Create a default time array for plotting.
        self.time = np.linspace(0, 20, num=10000)
        # Sum up all the individual sinusoidals to create a composed signal.
        for signal in self.sinusoidals:
            self.signal += signal
        # Store the mixed signal without noise.
        self.mixed_signal_with_noise = self.signal
        # Clear the previous plots and plot the new composed signal.
        self.Channel1Widget.clear()
        self.Channel2Widget.clear()
        self.Channel1Widget.plot(self.time, self.signal, pen=pg.mkPen(color='b'))
        # Apply the standard plot settings for the composed signal.
        self.plot_settings(self.Channel1Widget, self.time, self.signal, 'Composed Signal')


    def update_sampling_slider_max(self):
        max_frequency = 0
        # Iterate through each item in the SinusoidalCombobox
        for index in range(self.SinusoidalCombobox.count()):
            item_text = self.SinusoidalCombobox.itemText(index)
            # Extract the frequency value from the item text
            frequency = float(item_text.split(" - Frequency: ")[-1].replace("Hz", ""))
            if frequency > max_frequency:
                max_frequency = frequency

        self.fmax = max_frequency

        # Set the maximum value of the SamplingSlider
        slider_max_value = 4 * floor(self.fmax)
        self.SamplingSlider.setMaximum(slider_max_value)

        # Optionally, set the slider to its maximum value
        self.SamplingSlider.setValue(slider_max_value)




# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Remove The Sinusoidal Signals From The Original Signals
    # -------------------------------------------------------
    def remove_sinusoidal(self):
        selected_item_index = self.SinusoidalCombobox.currentIndex()

        if selected_item_index >= 0:
            # Remove the selected item from the ComboBox
            self.SinusoidalCombobox.removeItem(selected_item_index)
            # Recalculate the maximum frequency and update the slider's maximum
            self.update_sampling_slider_max()


            # Check if there's a corresponding signal to remove
            if selected_item_index < len(self.sinusoidals):
                # Remove the selected signal from the list
                del self.sinusoidals[selected_item_index]

                # Check if there are remaining sinusoidal signals
                if self.sinusoidals:
                    # Recalculate the mixed signal if there are remaining sinusoidals
                    self.update_mixed_signal()
                else:
                    # Clear the plot if there are no remaining sinusoidals
                    self.clear_handler()


# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Adding Noise on the Signals
    # ---------------------------
    def add_noise(self, snr_value):
        #Add Gaussian noise to the signal based on the provided SNR (Signal-to-Noise Ratio).
        # Calculate the mean and standard deviation for the Gaussian noise
        mean = 0
        std = np.sqrt(np.var(self.signal) / (10 ** (snr_value / 10)))
        
        # Get the number of samples in the signal
        num_samples = len(self.time)
        
        # Generate the Gaussian noise
        noise = np.random.normal(mean, std, size=num_samples)
        
        # Add the generated noise to the signal
        self.mixed_signal_with_noise = self.signal + noise
        
        # Clear previous plots and plot the new noisy signal
        self.Channel1Widget.clear()
        self.Channel1Widget.plot(self.time, self.mixed_signal_with_noise, pen=pg.mkPen(color='r', width=2))
        
        # Update plot settings and sampled points
        self.plot_settings(self.Channel1Widget, self.time, self.mixed_signal_with_noise, 'Mixed Signal with Noise')
        self.update_sampled_points()



# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Adjust the sampled points by starting a timer.
    # ----------------------------------------------
    def adjust_sampled_points(self):
        # Start the sampling timer
        self.sampling_timer.start()



# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Compute the sinc function for a given time array t.
    # --------------------------------------------------
    def sinc_function(self, t):
        # A threshold to handle the singularity at t=0
        threshold = 1e-6
        # Compute the sinc function
        # Use np.where to handle the singularity and avoid division by zero
        return np.where(np.abs(t) < threshold, 1.0, np.divide(np.sin(np.pi * t), np.pi * t, out=np.ones_like(t), where=np.abs(t)!=0))




# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # SINC Sinterpolation Function
    # ----------------------------
    # Reconstruct the signal using sinc interpolation based on resampled points.
    def sinc_interpolation(self, time, resampled_time, resampled_signal):
        # Calculate the sampling period from the resampled time data.
        T = resampled_time[1] - resampled_time[0]
        # Initialize the reconstructed signal to zeros.
        reconstructed_signal = np.zeros_like(time)
        # Determine the sampling frequency.
        sampling_frequency = 1.0 / T
        # Perform the sinc interpolation for each resampled point across the original time array.
        for n, (sample_t, sample_val) in enumerate(zip(resampled_time, resampled_signal)):
            # Accumulate the scaled sinc function for each sample over the signal.
            reconstructed_signal += sample_val * self.sinc_function((time - sample_t) * sampling_frequency)
        # Return the reconstructed signal after complete interpolation.
        return reconstructed_signal




# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Update All Sampling Points on Each Adding Signal
    # ------------------------------------------------
    # Recalculate and plot resampled points based on the new sampling frequency from the slider.
    def update_sampled_points(self):
        # Retrieve the new sample frequency from the slider
        new_sample_frequency = self.SamplingSlider.value()
        try:
            # Generate the new resampled time points
            resampled_time = np.arange(self.time[0], self.time[-1], 1 / new_sample_frequency)
        except:
            print(f"Trivial Error ... Don't Care - ErrorNew Sample Frequency Value is {new_sample_frequency}")

        # Create an interpolator function for the original signal with noise.
        interpolator = interp1d(self.time, self.mixed_signal_with_noise, kind='linear', fill_value="extrapolate")
        try:
            # Obtain the new resampled signal values at the resampled time points.
            resampled_signal = interpolator(resampled_time)
        except:
            print(f"Trivial Error ... Don't Care - Resampled Signal Value is 0")

        # Clear the previous plots and plot the resampled signal on Channel 2.
        self.Channel2Widget.clear()
        self.Channel1Widget.clear()
        try:
            # Plot resampled points as green dots on Channel 2.
            self.Channel1Widget.plot(self.time, self.mixed_signal_with_noise, pen=pg.mkPen(color='r', width=2))
            self.Channel1Widget.plot(resampled_time, resampled_signal, symbol='o', pen=None, symbolBrush=pg.mkBrush(color='g'), symbolSize=4)
            # Apply plot settings with the title 'Resampled Signal'.
            self.plot_settings(self.Channel2Widget, resampled_time, self.signal, "Resampled Signal")
            # Link the X and Y axis of Channel 1 and 2.
            self.Channel1Widget.setXLink(self.Channel2Widget)
            self.Channel1Widget.setYLink(self.Channel2Widget)

            # Use sinc interpolation to reconstruct the signal from the resampled points.
            reconstructed_signal = self.sinc_interpolation(self.time, resampled_time, resampled_signal)
            # Calculate the difference between the original and reconstructed signals.
            subtracted_signal = self.signal - reconstructed_signal

            # Clear Channel 3 and plot the subtracted signal.
            self.Channel3Widget.clear()
            self.Channel3Widget.plot(self.time, subtracted_signal, pen=pg.mkPen('r'))
            # Set the plot settings for the subtracted signal with title 'Subtracted Signal'.
            self.plot_settings(self.Channel3Widget, self.time, self.signal, "Subtracted Signal")
            # Link the X and Y axis of Channel 2 and 3.
            self.Channel2Widget.setXLink(self.Channel3Widget)
            self.Channel2Widget.setYLink(self.Channel3Widget)

            # Overlay the reconstructed signal on Channel 2 with a semi-transparent red pen.
            self.Channel2Widget.plot(self.time, reconstructed_signal, pen=pg.mkPen(color=(255, 0, 0, 100)))
            # Return the resampled time and signal for further processing if needed.
            return resampled_time, resampled_signal
        except:
            print(f"Trivial Error ... Don't Care - Resampled Time Value is 0")

        

        
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Sampling Slider Value Adjustment
    # --------------------------------    
    def set_sampling_slider_value(self, value):
        try:
            # Retrieve the maximum allowed frequency value
            max_allowed_frequency = 4 * floor(self.fmax)

            int_value = int(value)
            if int_value > max_allowed_frequency:
                # If the value exceeds the max allowed, reset to the max allowed frequency
                self.SamplingFrequencyEdit.setText(str(max_allowed_frequency))
                self.SamplingSlider.setValue(max_allowed_frequency)
            else:
                # Otherwise, set the slider value to the provided value
                self.SamplingSlider.setValue(int_value)
        except ValueError:
            # Handle the error, maybe by setting a default value or logging an error message
            print(f"Error: Invalid input in SamplingFrequencyEdit - {value}")
            # self.SamplingFrequencyEdit.setText(str(1))  # Resetting to some default value
            # self.SamplingSlider.setValue(1)



# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Connect The Sampling Slider to the Frequency
    # --------------------------------------------
    def connect_sampling_widgets(self):
        # Connect the value changed signal of the sampling slider to the appropriate slot.
        self.SamplingSlider.valueChanged.connect(self.set_sampling_edit_value)
        # Connect the text changed signal of the sampling frequency edit box to the appropriate slot.
        self.SamplingFrequencyEdit.textChanged.connect(self.set_sampling_slider_value)



    def nyquist(self):
        # If the Nyquist checkbox is checked, adjust the sampling slider's properties accordingly.
        if self.NyquistCheckbox.isChecked():
            # Set the minimum value of the slider to 0.
            self.SamplingSlider.setMinimum(0)
            # Set the maximum value of the slider to four times the maximum frequency present in the signal.
            self.SamplingSlider.setMaximum(4 * floor(self.fmax))
            # Set the step size of the slider to the floor value of the maximum frequency.
            self.SamplingSlider.setSingleStep(floor(self.fmax))
        else:
            # If the Nyquist checkbox is not checked, set default slider properties.
            self.SamplingSlider.setMinimum(0)
            self.SamplingSlider.setMaximum(1000)
            self.SamplingSlider.setSingleStep(50)



# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------



    # Edit The Frequency Edit Based on The Slider
    # -------------------------------------------
    def set_sampling_edit_value(self, value):
        # Update the sampling frequency edit text field to reflect the slider's value.
        self.SamplingFrequencyEdit.setText(str(value))





# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------





class SignalPreviewDialog(QDialog):
    """
    A dialog for previewing and adding sinusoidal waves to a signal plot.
    It allows the user to input parameters for a sinusoidal wave, preview the result,
    and then add the wave to a list of sinusoidals which updates the main plot.
    """


    def __init__(self, parent, existing_time, existing_signal, flag, sinusoidals):
        # Initialize the dialog, set up the UI and plotting area, and connect signals to slots.
        super(SignalPreviewDialog, self).__init__(parent)
        uic.loadUi("./Signal_Preview.ui", self)
        self.existing_time = existing_time
        self.existing_signal = existing_signal
        self.flag = flag
        if self.flag:
            self.SignalPreviewGraph.plot(existing_time, existing_signal, pen=pg.mkPen('b'))
        self.FrequencyEdit.textChanged.connect(self.add_sinusoidal_dialog)
        self.AmplitudeEdit.textChanged.connect(self.add_sinusoidal_dialog)
        self.AddButton.clicked.connect(lambda: self.add_signal(sinusoidals))
        self.PlotSlider.valueChanged.connect(self.slider_moved)




    def slider_moved(self, value):
        # Adjust the visible range of the signal preview graph based on the slider's value.
        visible_range = value / 100 * 20
        self.SignalPreviewGraph.setXRange(0, 0 + visible_range, padding=0)




    def add_signal(self, sinusoidals):
        # Add the created sinusoidal wave to the list and update the combobox in the main window.
        sinusoidals.append(self.sinusoidal_wave)
        self.parent().SinusoidalCombobox.addItem(
            f"Sinusoidal {len(sinusoidals)} - Amplitude: {self.amplitude} - Frequency: {self.frequency}Hz"
        )
        if self.frequency > self.parent().fmax:
            self.parent().fmax = self.frequency
        self.parent().update_sampling_slider_max() 
        self.close()


    
    



    def add_sinusoidal_dialog(self,):
        # Update the signal preview graph with a new sinusoidal wave based on input amplitude and frequency.
        amplitude_text = self.AmplitudeEdit.text()
        frequency_text = self.FrequencyEdit.text()

        # Check if amplitude and frequency fields are not empty
        if not amplitude_text or not frequency_text:
            # Handle this case, e.g., show an error message or return early
            return

        amplitude = float(amplitude_text)
        frequency = float(frequency_text)

        if frequency == 0:
            return

        self.time = np.linspace(0, 20, num=10000)
        self.sinusoidal_wave = amplitude * np.sin(2 * np.pi * frequency * self.time)
        if not self.flag:
            self.existing_signal = np.zeros_like(self.time)
            self.SignalPreviewGraph.setLimits(xMin = 0, xMax = 20, yMin = - amplitude - 0.2, yMax = amplitude + 0.2)
        else:
            self.SignalPreviewGraph.setLimits(xMin = 0, xMax = 20,\
                            yMin = min( - amplitude - 0.2, np.min(self.existing_signal) - 0.2 - amplitude), yMax = max(amplitude + 0.2, np.max(self.existing_signal + 0.2 + amplitude)))
        self.signal = self.existing_signal + self.sinusoidal_wave

        self.SignalPreviewGraph.clear()
        self.SignalPreviewGraph.plot(self.time, self.signal, pen = pg.mkPen(color = 'b'))
        self.amplitude, self.frequency = amplitude, frequency










# RUN THE MAIN APPLICATION
# ------------------------
def main():
    app = QApplication([])
    window = SignalStudio()
    app.exec_()

if __name__ == "__main__":
    main()
