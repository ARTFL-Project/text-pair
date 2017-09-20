<template>
  <div id="app">
    <div class="container-fluid">
      <search-form></search-form>
      <router-view></router-view>
    </div>
  </div>
</template>

<script>
import searchForm from './components/searchForm'

export default {
  name: 'app',
  components: {
    searchForm
  }
}
</script>

<style>
#app {
  font-family: 'Open Sans', sans-serif;
}

.shadow-1 {
  box-shadow: 0 2px 5px 0 rgba(0, 0, 0, .16), 0 2px 10px 0 rgba(0, 0, 0, .12);
}
</style>
