document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM content loaded.");

    const socket = io(); // Connect to the SocketIO server
    console.log("Connected to the SocketIO server.");

    // Get canvas elements and audio context
    const canvas = document.getElementById("audioVisualizer");
    const canvasCtx = canvas.getContext("2d");
    const canvasMFCC = document.getElementById("mfccVisualizer");
    const canvasMFCCCtx = canvasMFCC.getContext("2d");

    // Set canvas dimensions
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    canvasMFCC.width = canvasMFCC.offsetWidth;
    canvasMFCC.height = canvasMFCC.offsetHeight;

    let isPlaying = false;

    // Variables to store frequency and magnitude spectrum data
    let frequencies = [];
    let magnitudeSpectrum = [];

    // Function to draw waveform from spectrum data
    function drawWaveformFromSpectrum(frequencies, magnitudeSpectrum) {
        console.log("Drawing waveform from spectrum...");

        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

        if (frequencies && magnitudeSpectrum) {
            const bufferLength = frequencies.length;
            const sliceWidth = canvas.width / bufferLength;
            let x = 0;

            canvasCtx.lineWidth = 2;
            canvasCtx.strokeStyle = "rgb(0, 0, 0)";
            canvasCtx.beginPath();

            for (let i = 0; i < bufferLength; i++) {
                const v = magnitudeSpectrum[i] / 128.0; // Normalize magnitude spectrum
                const y = (v * canvas.height) / 2;

                if (i === 0) {
                    canvasCtx.moveTo(x, y);
                } else {
                    canvasCtx.lineTo(x, y);
                }

                x += sliceWidth;
            }

            canvasCtx.lineTo(canvas.width, canvas.height / 2);
            canvasCtx.stroke();
        }
    }

    // Function to draw MFCC data as a bar graph
    function drawMFCCBarGraph(mfccData) {
        console.log("Drawing MFCC bar graph...");

        canvasMFCCCtx.clearRect(0, 0, canvasMFCC.width, canvasMFCC.height);

        if (mfccData) {
            const numCoefficients = mfccData.length;
            const barWidth = canvasMFCC.width / numCoefficients;
            const scaleFactor = canvasMFCC.height / 2; // Scale to half height for better visualization
            const offsetY = canvasMFCC.height / 2; // Offset the graph vertically to the middle

            for (let i = 0; i < numCoefficients; i++) {
                const coefficientValue = mfccData[i] * scaleFactor;
                const x = i * barWidth;
                const y = offsetY - coefficientValue;

                canvasMFCCCtx.fillStyle = "rgb(0, 0, 255)";
                canvasMFCCCtx.fillRect(x, y, barWidth, coefficientValue);
            }
        }
    }

    // Function to update the average frequency and magnitude below the spectrograms
    function updateFrequencyMagnitudeInfo() {
        const averageFrequencyElement = document.getElementById("averageFrequency");
        const averageMagnitudeElement = document.getElementById("averageMagnitude");
        const maxDurationElement = document.getElementById("maxDuration");

        const averageFrequency = frequencies.length > 0 ? frequencies.reduce((acc, val) => acc + val, 0) / frequencies.length : 0;
        const averageMagnitude = magnitudeSpectrum.length > 0 ? magnitudeSpectrum.reduce((acc, val) => acc + val, 0) / magnitudeSpectrum.length : 0;
        const maxDuration = maxDurationElement.textContent;

        averageFrequencyElement.textContent = `Average Frequency: ${averageFrequency.toFixed(8)} Hz`;
        averageMagnitudeElement.textContent = `Average Magnitude: ${averageMagnitude.toFixed(8)}`;
        maxDurationElement.textContent = `Maximum Duration: ${maxDuration} seconds`;
    }

// Function to display distress details in a text card and make it flash
function showDistressDetails(distressType) {
    // Define an object with distress details based on the distress types
    const distressDetails = {
      clucking: {
        title: "Clucking Distress",
        description: "Clucking is a distress behavior observed in certain situations...",
        // Add more properties as needed
      },
      clacking: {
        title: "Clacking Distress",
        description: "Clacking is a distress behavior characterized by rapid sounds...",
        // Add more properties as needed
      },
      alert_call: {
        title: "Alert Call Distress",
        description: "An alert call is a vocalization used by birds to warn others...",
        // Add more properties as needed
      },
      brooding: {
        title: "Brooding Distress",
        description: "Brooding is a distress behavior where a bird exhibits...",
        // Add more properties as needed
      },
      crying: {
        title: "Crying Distress",
        description: "Crying is a distress vocalization that indicates...",
        // Add more properties as needed
      },
      squawking: {
        title: "Squawking Distress",
        description: "Squawking is a distress vocalization that is loud and intense...",
        // Add more properties as needed
      },
      // Add details for other distress types
    };
  
   
  // Get the text card container element
  const textCardContainer = document.getElementById("textCardContainer");

  // Check if the distress type is in the distressDetails object
  if (distressType in distressDetails) {
    // Create a new text card element and populate it with the distress details
    const textCard = document.createElement("div");
    textCard.className = "textCard";
    textCard.innerHTML = `
      <h2>${distressDetails[distressType].title}</h2>
      <p>${distressDetails[distressType].description}</p>
      <!-- Add more HTML elements to show additional details -->
    `;

    // Clear the container and add the text card
    textCardContainer.innerHTML = "";
    textCardContainer.appendChild(textCard);

    // Show the text card container
    textCardContainer.style.display = "block";

    // Optional: Flash the text card
    let isHidden = false;
    const flashingInterval = setInterval(() => {
      if (isHidden) {
        textCardContainer.style.display = "block";
      } else {
        textCardContainer.style.display = "none";
      }
      isHidden = !isHidden;
    }, 500); // Flashing interval in milliseconds (e.g., 500ms)

    // Optional: Stop the flashing after a certain duration (e.g., 5 seconds)
    setTimeout(() => {
      clearInterval(flashingInterval);
      textCardContainer.style.display = "none"; // Hide the text card container after the flashing stops
    }, 5000); // Duration in milliseconds (e.g., 5000ms = 5 seconds)
  }
}
    // Event listener to receive MFCC data from the backend
    socket.on("mfcc_data", (data) => {
        // Process the MFCC data and draw the bar graph
        const mfccData = data.mfcc_data;
        console.log("Received MFCC data:", mfccData);
        drawMFCCBarGraph(mfccData);
    });

// Event listener to receive spectrum data from the backend and visualize it as a waveform
socket.on("spectrum_data", (data) => {
    // Update the global variables instead of redeclaring them locally
    frequencies = data.frequencies;
    magnitudeSpectrum = data.magnitude_spectrum;
    console.log("Received frequencies:", frequencies);
    console.log("Received magnitude spectrum:", magnitudeSpectrum);

    // Draw the waveform using the spectrum data
    drawWaveformFromSpectrum(frequencies, magnitudeSpectrum);

    // Plot frequency spectogram using Plotly
    const traceFrequency = {
        x: new Array(frequencies.length).fill(data.time),
        y: frequencies,
        mode: "lines",
        type: "scatter",
    };

    const layoutFrequency = {
        title: "Frequency Spectogram",
        xaxis: { title: "Time (s)" },
        yaxis: { title: "Frequency (Hz)" },
    };

    Plotly.newPlot("spectrumGraph", [traceFrequency], layoutFrequency);

    // Plot magnitude spectogram using Plotly
    const traceMagnitude = {
        x: new Array(magnitudeSpectrum.length).fill(data.time),
        y: magnitudeSpectrum,
        mode: "lines",
        type: "scatter",
    };

    const layoutMagnitude = {
        title: "Magnitude Spectogram",
        xaxis: { title: "Time (s)" },
        yaxis: { title: "Magnitude" },
    };

    Plotly.newPlot("spectrumGraphMagnitude", [traceMagnitude], layoutMagnitude);

    // Update frequency, magnitude, and duration info
    updateFrequencyMagnitudeInfo();
});


    // Event listener to receive classification result from the backend
socket.on("classification_result", (data) => {
    console.log("Received classification result:", data);
  
    const distressType = data.distress_type;
    const classificationResultElement = document.getElementById("classificationResult");
    classificationResultElement.textContent = `Classification Result: ${distressType}`;
  
    // Call the function to show the distress details in a text card
    showDistressDetails(distressType);
  });
  

    // Event listener to receive maximum duration from the backend
    socket.on("max_duration", (data) => {
        const maxDuration = data.duration;
        console.log("Received max duration:", maxDuration);
        document.getElementById("maxDuration").textContent = maxDuration.toFixed(2);
        updateFrequencyMagnitudeInfo();
    });

    // Start button click handler
const startButton = document.getElementById("startButton");
startButton.addEventListener("click", () => {
    if (!isPlaying) {
        const confirmationMessage = "Are you sure you want to start the Chic Device?";{
            console.log("Starting audio processing...");
            socket.emit("start_audio_processing"); // Send a message to the backend to start processing audio
            isPlaying = true;
        }
    } else {
        const confirmationMessage = "Are you sure you want to stop the Chic Device?";
        if (confirm(confirmationMessage)) {
            console.log("Stopping audio processing...");
            socket.emit("stop_audio_processing"); // Send a message to the backend to stop processing audio
            isPlaying = false;
        }
    }
});
});
