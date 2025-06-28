<script>
  let keyword = "";
  let date = "";
  let collection = "";

  const handleSearch = () => {
    console.log("Searching for:", { keyword, date, collection });
    // You can replace this with an API call
  };

</script>

<header class="header"> 
  <h4>Heritage Metadata Explorer</h4>
  <nav>
    <a href="https://digitalcollections.rice.edu/white-house-scientist-and-science-policy-dynamic-digital-archive" target="_blank">Rice Archives</a>
    <a href="/process">Process Data</a>
  </nav>
</header>

<main>
  <section class="search-panel">
    <input type="text" placeholder="Search by keyword..." bind:value={keyword} />
    <input type="date" bind:value={date} />
    <input type="text" placeholder="Search by collection..." bind:value={collection} />
    <button on:click={handleSearch}>Search</button>
  </section>
</main>

<style>
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #333;
    color: white;
    padding: 1rem;
  }

  nav a {
    color: white;
    margin-left: 1rem;
    text-decoration: none;
  }

  .search-panel {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 2rem;
    max-width: 400px;
    margin: auto;
  }

  input, button {
    padding: 0.5rem;
    font-size: 1rem;
  }

  button {
    background-color: #ff3e00;
    color: white;
    border: none;
    cursor: pointer;
  }
</style>
