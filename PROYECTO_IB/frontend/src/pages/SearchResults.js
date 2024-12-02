import React, { useState, useEffect } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { FaSearch } from "react-icons/fa";
import { ClipLoader } from "react-spinners";

function SearchResults() {
  const [searchParams] = useSearchParams();
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState("tfidf");
  const [results, setResults] = useState([]);
  const [metrics, setMetrics] = useState({
    precision: null,
    recall: null,
    f1_score: null,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const resultsPerPage = 10;

  const navigate = useNavigate();

  useEffect(() => {
    const queryParam = searchParams.get("query") || "";
    const methodParam = searchParams.get("method") || "tfidf";
    setQuery(queryParam);
    setMethod(methodParam);
    if (queryParam) {
      handleSearch(queryParam, methodParam);
    }
  }, [searchParams]);

  const handleSearch = async (searchQuery, searchMethod) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("http://127.0.0.1:5000/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: searchQuery, method: searchMethod }),
      });

      if (!response.ok) {
        throw new Error("Error al realizar la búsqueda");
      }

      const data = await response.json();
      console.log("Datos recibidos del backend:", data);

      if (data && data.resultados) {
        setResults(data.resultados || []);
        setMetrics({
          precision: data.precision || null,
          recall: data.recall || null,
          f1_score: data.f1_score || null,
        });
        console.log("Métricas asignadas al estado:", {
          precision: data.precision,
          recall: data.recall,
          f1_score: data.f1_score,
        });
      } else {
        setResults([]);
        setMetrics({
          precision: null,
          recall: null,
          f1_score: null,
        });
      }

      setCurrentPage(1);
    } catch (err) {
      setError(err.message);
      setResults([]);
      setMetrics({
        precision: null,
        recall: null,
        f1_score: null,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const indexOfLastResult = currentPage * resultsPerPage;
  const indexOfFirstResult = indexOfLastResult - resultsPerPage;
  const currentResults = Array.isArray(results)
    ? results.slice(indexOfFirstResult, indexOfLastResult)
    : [];

  const totalPages = Math.ceil(results.length / resultsPerPage);

  const handlePageChange = (pageNumber) => {
    setCurrentPage(pageNumber);
  };

  const handleResultClick = (result) => {
    navigate("/content", { state: result });
  };

  const renderPaginationButtons = () => {
    const paginationButtons = [];
    const maxVisibleButtons = 4;

    if (currentPage > 1) {
      paginationButtons.push(
        <button
          key="prev"
          onClick={() => handlePageChange(currentPage - 1)}
          style={paginationButtonStyle}
        >
          &lt; Anterior
        </button>
      );
    }

    for (let i = 1; i <= totalPages; i++) {
      if (
        i === 1 ||
        i === totalPages ||
        (i >= currentPage - 1 && i <= currentPage + 1)
      ) {
        paginationButtons.push(
          <button
            key={i}
            onClick={() => handlePageChange(i)}
            style={
              currentPage === i ? activeButtonStyle : paginationButtonStyle
            }
          >
            {i}
          </button>
        );
      } else if (i === currentPage - 2 || i === currentPage + 2) {
        paginationButtons.push(
          <span key={`ellipsis-${i}`} style={ellipsisStyle}>
            ...
          </span>
        );
      }
    }

    if (currentPage < totalPages) {
      paginationButtons.push(
        <button
          key="next"
          onClick={() => handlePageChange(currentPage + 1)}
          style={paginationButtonStyle}
        >
          Siguiente &gt;
        </button>
      );
    }

    return paginationButtons;
  };

  const paginationButtonStyle = {
    padding: "10px 15px",
    border: "1px solid #ddd",
    borderRadius: "5px",
    backgroundColor: "#ffffff",
    color: "#007bff",
    cursor: "pointer",
    fontSize: "1rem",
    minWidth: "50px",
    textAlign: "center",
  };

  const activeButtonStyle = {
    ...paginationButtonStyle,
    backgroundColor: "#007bff",
    color: "#ffffff",
    fontWeight: "bold",
  };

  const ellipsisStyle = {
    margin: "0 5px",
    color: "#888",
  };

  return (
    <div
      style={{
        fontFamily: "'Arial', sans-serif",
        backgroundColor: "#f8f9fa",
        minHeight: "100vh",
      }}
    >
      <nav
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "98%",
          height: "65px",
          zIndex: 1000,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "linear-gradient(to right, #00bcd4, #007bff)",
          color: "white",
          padding: "10px 20px",
          boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
        }}
      >
        <h1 style={{ fontSize: "1.5rem", margin: 0 }}>Reuters Search</h1>
        <div
          style={{
            display: "flex",
            flex: 1,
            margin: "0 20px",
            alignItems: "center",
            gap: "10px",
          }}
        >
          <div
            style={{
              position: "relative",
              flex: 1,
              display: "flex",
              alignItems: "center",
              backgroundColor: "#fff",
              borderRadius: "25px",
              boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
              padding: "1px 10px",
            }}
          >
            <FaSearch
              style={{
                position: "absolute",
                left: "15px",
                color: "#ccc",
                fontSize: "1.2rem",
              }}
            />
            <input
              type="text"
              placeholder="Search in Reuters..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  handleSearch(query, method);
                }
              }}
              style={{
                flex: 1,
                padding: "10px 30px",
                paddingLeft: "35px",
                borderRadius: "25px",
                border: "none",
                fontSize: "1rem",
                outline: "none",
              }}
            />
          </div>

          <button
            onClick={() => handleSearch(query, method)}
            style={{
              padding: "10px 20px",
              borderRadius: "25px",
              border: "none",
              backgroundColor: "#ffffff",
              color: "#007bff",
              cursor: "pointer",
              fontSize: "1rem",
              boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
            }}
          >
            Buscar
          </button>
        </div>
        <select
          id="method"
          value={method}
          onChange={(e) => {
            setMethod(e.target.value);
            handleSearch(query, e.target.value);
          }}
          style={{
            padding: "10px 15px",
            borderRadius: "5px",
            border: "1px solid #ddd",
            fontSize: "1rem",
            backgroundColor: "#ffffff",
            color: "#007bff",
          }}
        >
          <option value="tfidf">TF-IDF</option>
          <option value="bow">Bag of Words</option>
          <option value="word2vec">Word2Vec</option>
        </select>
      </nav>

      <div style={{ height: "70px" }}></div>

      <div style={{ maxWidth: "800px", margin: "0 auto" }}>
        

          <div
            style={{
              backgroundColor: "#e0f7fa",
              padding: "10px",
              borderRadius: "5px",
              marginBottom: "20px",
            }}
          >
            <h3>Métricas de Evaluación:</h3>
            <p><strong>Precisión:</strong> {metrics.precision !== null ? metrics.precision : "No disponible"}</p>
            <p><strong>Recall:</strong> {metrics.recall !== null ? metrics.recall : "No disponible"}</p>
            <p><strong>F1-Score:</strong> {metrics.f1_score !== null ? metrics.f1_score : "No disponible"}</p>
          </div>

        {isLoading && <ClipLoader color="#007bff" size={50} />}

        {error && <p>Error: {error}</p>}

        {results.length === 0 && !isLoading && !error && (
          <p>No se encontraron resultados</p>
        )}

        {currentResults.map((result, index) => (
          <div key={index} onClick={() => handleResultClick(result)}>
            <h3>{result.Titulo}</h3>
            <p>{result.Contenido}</p>
            <p>Similitud: {result.Similitud}</p>
          </div>
        ))}

        <div>{renderPaginationButtons()}</div>
      </div>
    </div>
  );
}

export default SearchResults;
