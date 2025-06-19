<script>
  let files = [];
  let ocrEngine = "Mistral";
  let itemizer = "Gemini";
  let metadataModel = "DeepSeek";

  let ocrText = "";
  let metadata = null;
  let processing = false;

  function handleFileUpload(event) {
    files = Array.from(event.target.files);
  }

  async function runOCR() {
    processing = true;
    // Simulate OCR processing
    await new Promise(resolve => setTimeout(resolve, 1500));
    ocrText = "This is a simulated OCR result from the uploaded documents.";
    processing = false;
  }

  async function itemizePDFs() {
    processing = true;
    // Simulate itemizing
    await new Promise(resolve => setTimeout(resolve, 1200));
    ocrText += "\n\n[Items extracted with " + itemizer + "]";
    processing = false;
  }

  async function extractMetadata() {
    processing = true;
    // Simulate metadata extraction
    await new Promise(resolve => setTimeout(resolve, 1800));
    metadata = {
      title: "Sample Heritage Artifact",
      date: "1887",
      description: "This artifact was extracted using " + metadataModel + ".",
      tags: ["archive", "historical", "metadata"]
    };
    processing = false;
  }
</script>

<style>
  .container {
    @apply max-w-4xl mx-auto p-6 space-y-6;
  }
  label {
    @apply font-semibold block mb-1;
  }
  select, input[type="file"] {
    @apply border p-2 w-full rounded mb-4;
  }
  button {
    @apply bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50;
  }
  .section {
    @apply border-t pt-6;
  }
  textarea {
    @apply w-full p-3 border rounded bg-gray-50;
    min-height: 150px;
  }
  pre {
    @apply bg-gray-100 p-4 rounded overflow-x-auto;
  }
</style>

<div class="container">
  <h1 class="text-2xl font-bold">📄 Process Heritage Documents</h1>

  <!-- File Upload -->
  <div>
    <label>Upload PDF or JPEG Files</label>
    <input type="file" accept="application/pdf,image/jpeg" multiple on:change={handleFileUpload} />
  </div>

  <!-- OCR Section -->
  <div class="section">
    <label for="ocrEngine">Choose OCR Engine</label>
    <select id="ocrEngine" bind:value={ocrEngine}>
      <option value="Mistral">Mistral</option>
      <option value="Textract">Textract</option>
    </select>
    <button on:click={runOCR} disabled={processing}>Run OCR</button>
  </div>

  <!-- OCR Output -->
  {#if ocrText}
    <div class="section">
      <label>OCR Output</label>
      <textarea readonly>{ocrText}</textarea>
    </div>
  {/if}

  <!-- Itemizing -->
  {#if ocrText}
    <div class="section">
      <label for="itemizer">Choose Itemizer Model</label>
      <select id="itemizer" bind:value={itemizer}>
        <option value="Gemini">Gemini</option>
        <option value="ChatGPT">ChatGPT</option>
      </select>
      <button on:click={itemizePDFs} disabled={processing}>Itemize PDFs</button>
    </div>
  {/if}

  <!-- Metadata Extraction -->
  {#if ocrText}
    <div class="section">
      <label for="metadataModel">Metadata Extraction Model</label>
      <select id="metadataModel" bind:value={metadataModel}>
        <option value="DeepSeek">DeepSeek</option>
        <option value="Claude">Claude</option>
        <option value="Gemini">Gemini</option>
        <option value="ChatGPT">ChatGPT</option>
      </select>
      <button on:click={extractMetadata} disabled={processing}>Extract Metadata</button>
    </div>
  {/if}

  <!-- Metadata Display -->
  {#if metadata}
    <div class="section">
      <label>Extracted Metadata</label>
      <pre>{JSON.stringify(metadata, null, 2)}</pre>
    </div>
  {/if}
</div>