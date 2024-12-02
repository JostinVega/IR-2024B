import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import SearchResults from "./pages/SearchResults";
import Content from "./pages/Content";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/results" element={<SearchResults />} />
        <Route path="/content" element={<Content />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
