document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const tabs = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const statusBadge = document.getElementById("statusBadge");
  const dropzone = document.getElementById("dropzone");
  const audioFileInput = document.getElementById("audioFileInput");
  const demoButtonsContainer = document.getElementById("demoButtonsContainer");
  const recBtn = document.getElementById("recBtn");
  const recTimer = document.getElementById("recTimer");
  const recHint = document.getElementById("recHint");
  
  const audioPlayer = document.getElementById("audioPlayer");
  const audioPlayerBox = document.getElementById("audioPlayerBox");
  const currentAudioName = document.getElementById("currentAudioName");
  const waveformCanvas = document.getElementById("waveformCanvas");
  
  const loadingOverlay = document.getElementById("loadingOverlay");
  const resultsContent = document.getElementById("resultsContent");
  const emptyResultsHint = document.getElementById("emptyResultsHint");
  
  const primaryResultCard = document.getElementById("primaryResultCard");
  const emotionHeroTitle = document.getElementById("emotionHeroTitle");
  const confidenceHero = document.getElementById("confidenceHero");
  const probList = document.getElementById("probList");
  
  const statDuration = document.getElementById("statDuration");
  const statSampleRate = document.getElementById("statSampleRate");
  const statRms = document.getElementById("statRms");
  const statZcr = document.getElementById("statZcr");

  // State
  let mediaRecorder = null;
  let audioChunks = [];
  let recordingInterval = null;
  let recordingSeconds = 0;
  let audioContext = null;
  let audioBuffer = null;

  // Initialize System Status
  async function initStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.model_trained) {
        statusBadge.innerHTML = `<span class="status-dot"></span> Model Ready (${data.model_name})`;
        statusBadge.className = "status-badge";
      } else {
        statusBadge.innerHTML = `<span class="status-dot"></span> Model Not Trained`;
        statusBadge.className = "status-badge offline";
      }

      // Render Demo Samples Chips
      if (data.demo_samples && data.demo_samples.length > 0) {
        demoButtonsContainer.innerHTML = "";
        data.demo_samples.forEach(sample => {
          const color = data.colors[sample] || "#6366f1";
          const btn = document.createElement("button");
          btn.className = "demo-chip";
          btn.id = `demo-btn-${sample}`;
          btn.innerHTML = `
            <span class="chip-dot" style="background: ${color}; box-shadow: 0 0 8px ${color}"></span>
            <span style="text-transform: capitalize;">${sample}</span>
          `;
          btn.addEventListener("click", () => runDemoPrediction(sample));
          demoButtonsContainer.appendChild(btn);
        });
      }
    } catch (err) {
      statusBadge.innerHTML = `<span class="status-dot"></span> Backend Offline`;
      statusBadge.className = "status-badge offline";
    }
  }

  // Tab switching
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tabContents.forEach(tc => tc.classList.remove("active"));
      tab.classList.add("active");
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.add("active");
    });
  });

  // Drag and drop setup
  ["dragenter", "dragover"].forEach(event => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(event => {
    dropzone.addEventListener(event, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileUpload(files[0]);
  });

  dropzone.addEventListener("click", () => audioFileInput.click());
  audioFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
  });

  // File Upload Handler
  async function handleFileUpload(file) {
    if (!file.type.includes("audio") && !file.name.endsWith(".wav") && !file.name.endsWith(".mp3")) {
      alert("Please upload a valid audio file (.wav, .mp3, .ogg)");
      return;
    }
    setAudioPlayerSrc(URL.createObjectURL(file), file.name);
    drawWaveformFromFile(file);

    const formData = new FormData();
    formData.append("file", file);

    await sendPredictionRequest("/api/predict/upload", {
      method: "POST",
      body: formData
    });
  }

  // Demo Sample Prediction
  async function runDemoPrediction(sampleName) {
    const audioUrl = `/demo_audio/${sampleName.toLowerCase()}.wav`;
    setAudioPlayerSrc(audioUrl, `RAVDESS Sample: ${sampleName}.wav`);
    drawWaveformFromUrl(audioUrl);

    await sendPredictionRequest(`/api/predict/sample/${sampleName}`, {
      method: "POST"
    });
  }

  // Microphone Recording Handler
  recBtn.addEventListener("click", toggleRecording);

  async function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording();
    } else {
      startRecording();
    }
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        try {
          const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
          // Convert browser WebM/Opus stream to authentic 16-bit PCM WAV
          const wavBlob = await convertBlobToWav(rawBlob);
          const audioUrl = URL.createObjectURL(wavBlob);
          setAudioPlayerSrc(audioUrl, "Microphone Voice Recording");
          drawWaveformFromFile(wavBlob);

          const formData = new FormData();
          formData.append("file", wavBlob, "voice_recording.wav");
          await sendPredictionRequest("/api/predict/upload", {
            method: "POST",
            body: formData
          });
        } catch (convErr) {
          console.error("Audio processing/conversion error:", convErr);
          alert("Audio processing error: " + convErr.message);
        }
      };

      mediaRecorder.start();
      recBtn.classList.add("recording");
      recHint.textContent = "Click to stop recording";
      recordingSeconds = 0;
      recTimer.textContent = "00:00";
      recordingInterval = setInterval(() => {
        recordingSeconds++;
        const mins = String(Math.floor(recordingSeconds / 60)).padStart(2, "0");
        const secs = String(recordingSeconds % 60).padStart(2, "0");
        recTimer.textContent = `${mins}:${secs}`;
      }, 1000);
    } catch (err) {
      alert("Unable to access microphone. Please grant permission: " + err.message);
    }
  }

  function stopRecording() {
    if (mediaRecorder) {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    recBtn.classList.remove("recording");
    recHint.textContent = "Click to start recording voice note";
    clearInterval(recordingInterval);
  }

  // API Call Wrapper
  async function sendPredictionRequest(url, options) {
    loadingOverlay.style.display = "block";
    resultsContent.style.display = "none";
    emptyResultsHint.style.display = "none";

    try {
      const res = await fetch(url, options);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }
      const data = await res.json();
      renderResults(data);
    } catch (err) {
      alert("Error: " + err.message);
      emptyResultsHint.style.display = "block";
    } finally {
      loadingOverlay.style.display = "none";
    }
  }

  // Render Prediction Results
  function renderResults(data) {
    emptyResultsHint.style.display = "none";
    resultsContent.style.display = "flex";

    const topEmotion = data.predicted_emotion;
    const topPercent = data.confidence_percent;
    const topColor = data.color || "#6366f1";

    // Primary Card
    primaryResultCard.style.borderColor = topColor;
    primaryResultCard.style.boxShadow = `0 0 25px ${topColor}40`;
    emotionHeroTitle.textContent = topEmotion;
    emotionHeroTitle.style.color = topColor;
    confidenceHero.textContent = `Confidence: ${topPercent}%`;

    // Probability Bars
    probList.innerHTML = "";
    data.ranking.forEach(item => {
      const itemEl = document.createElement("div");
      itemEl.className = "prob-item";
      itemEl.innerHTML = `
        <div class="prob-label-row">
          <span style="text-transform: capitalize; color: #f8fafc;">${item.emotion}</span>
          <span style="color: #94a3b8; font-weight: 700;">${item.percentage}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: 0%; background: ${item.color};"></div>
        </div>
      `;
      probList.appendChild(itemEl);

      // Animate width
      setTimeout(() => {
        const fill = itemEl.querySelector(".prob-fill");
        if (fill) fill.style.width = `${item.percentage}%`;
      }, 50);
    });

    // Audio Signal Stats
    if (data.audio_stats) {
      statDuration.textContent = `${data.audio_stats.duration_sec}s`;
      statSampleRate.textContent = `${data.audio_stats.sample_rate} Hz`;
      statRms.textContent = data.audio_stats.rms_energy;
      statZcr.textContent = data.audio_stats.zero_crossing_rate;
    }
  }

  function setAudioPlayerSrc(src, name) {
    audioPlayer.src = src;
    audioPlayerBox.style.display = "block";
    currentAudioName.textContent = name;
  }

  // Waveform Drawing
  async function drawWaveformFromFile(fileOrBlob) {
    const arrayBuffer = await fileOrBlob.arrayBuffer();
    drawWaveformBuffer(arrayBuffer);
  }

  async function drawWaveformFromUrl(url) {
    const res = await fetch(url);
    const arrayBuffer = await res.arrayBuffer();
    drawWaveformBuffer(arrayBuffer);
  }

  async function drawWaveformBuffer(arrayBuffer) {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    try {
      const decodedData = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      const channelData = decodedData.getChannelData(0);
      drawCanvasWaveform(channelData);
    } catch (e) {
      console.warn("Waveform decode warning:", e);
    }
  }

  function drawCanvasWaveform(samples) {
    const canvas = waveformCanvas;
    const ctx = canvas.getContext("2d");
    const width = (canvas.width = canvas.offsetWidth * 2);
    const height = (canvas.height = canvas.offsetHeight * 2);

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, 0, width, height);

    const step = Math.ceil(samples.length / width);
    const amp = height / 2;

    ctx.fillStyle = "#6366f1";
    for (let i = 0; i < width; i++) {
      let min = 1.0;
      let max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = samples[i * step + j];
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      const y1 = (1 + min) * amp;
      const y2 = Math.max((1 + max) * amp, y1 + 1);
      ctx.fillRect(i, y1, 1, y2 - y1);
    }
  }

  // Convert arbitrary audio Blob to standard 16-bit PCM WAV Blob
  async function convertBlobToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await decodeCtx.decodeAudioData(arrayBuffer);

    const numOfChan = decoded.numberOfChannels;
    const sampleRate = decoded.sampleRate;
    const length = decoded.length * numOfChan * 2 + 44;
    const outBuffer = new ArrayBuffer(length);
    const view = new DataView(outBuffer);

    // RIFF identifier
    writeUTFBytes(view, 0, 'RIFF');
    // file length
    view.setUint32(4, 36 + decoded.length * numOfChan * 2, true);
    // RIFF type
    writeUTFBytes(view, 8, 'WAVE');
    // format chunk identifier
    writeUTFBytes(view, 12, 'fmt ');
    // format chunk length
    view.setUint32(16, 16, true);
    // sample format (1 = PCM)
    view.setUint16(20, 1, true);
    // channel count
    view.setUint16(22, numOfChan, true);
    // sample rate
    view.setUint32(24, sampleRate, true);
    // byte rate (sample rate * block align)
    view.setUint32(28, sampleRate * numOfChan * 2, true);
    // block align (channel count * bytes per sample)
    view.setUint16(32, numOfChan * 2, true);
    // bits per sample
    view.setUint16(34, 16, true);
    // data chunk identifier
    writeUTFBytes(view, 36, 'data');
    // data chunk length
    view.setUint32(40, decoded.length * numOfChan * 2, true);

    // write interleaved 16-bit PCM samples
    const channels = [];
    for (let i = 0; i < numOfChan; i++) {
      channels.push(decoded.getChannelData(i));
    }

    let offset = 44;
    for (let i = 0; i < decoded.length; i++) {
      for (let ch = 0; ch < numOfChan; ch++) {
        let sample = channels[ch][i];
        sample = Math.max(-1, Math.min(1, sample));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
      }
    }

    return new Blob([outBuffer], { type: "audio/wav" });
  }

  function writeUTFBytes(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  // Run on load
  initStatus();
});
