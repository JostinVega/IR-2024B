import React, { useState } from "react"; 
import { useNavigate } from "react-router-dom"; // Importa el hook useNavigate para navegar entre rutas
import reutersLogo from "../img/reuters-logo.png"; // Asegúrate de importar correctamente la imagen del logo
import { FaSearch } from "react-icons/fa"; // Importa el ícono de lupa desde react-icons

function Home() {
  // Definición de los estados que maneja este componente
  const [query, setQuery] = useState(""); // Estado para almacenar el texto de la búsqueda
  const [method, setMethod] = useState("tfidf"); // Estado para seleccionar el método de búsqueda, por defecto "tfidf"
  const navigate = useNavigate(); // Hook para manejar la navegación entre páginas

  // Función que maneja la acción de búsqueda, redirige a la página de resultados
  const handleSearch = () => {
    navigate(`/results?query=${encodeURIComponent(query)}&method=${method}`);
  };

  // Variable para el color de enfoque al pasar el mouse o hacer foco sobre ciertos elementos
  const hoverFocusColor = "#007bff"; 

  // Manejar el evento de presionar Enter para realizar la búsqueda
  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        height: "100vh", 
        fontFamily: "'Arial', sans-serif", 
        backgroundColor: "#f8f9fa", 
        position: "relative", 
      }}
    >
      {/* Logo */}
      <img
        src={reutersLogo}
        alt="Reuters Logo"
        style={{
          position: "absolute", 
          top: "-10px", 
          left: "20px", 
          width: "160px", 
          height: "auto", 
          opacity: "0.8", 
        }}
      />

      {/* Título principal */}
      <div
        style={{
          position: "relative",
          textAlign: "center",
          marginBottom: "20px",
          marginTop: "40px", 
        }}
      >
        <h1
          style={{
            fontSize: "4rem", 
            color: "#0056b3", 
            fontWeight: "bold", 
            textShadow: "2px 2px 5px rgba(0, 0, 0, 0.1)", 
            position: "relative",
            letterSpacing: "1px", 
            margin: "0", 
            paddingBottom: "30px", 
          }}
        >
          Reuters Search
        </h1>
      </div>

      {/* Contenedor para la barra de búsqueda y selector */}
      <div
        style={{
          display: "flex",
          justifyContent: "center", 
          alignItems: "center", 
          gap: "10px", 
          position: "relative",
          width: "100%", 
        }}
      >
        {/* Barra de búsqueda con lupa */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)", 
            borderRadius: "25px", 
            backgroundColor: "#fff", 
            padding: "10px 20px", 
            width: "600px", 
            transition: "box-shadow 0.3s ease", 
          }}
          onMouseOver={(e) => (e.currentTarget.style.boxShadow = "0 6px 15px rgba(0, 0, 0, 0.15)")} 
          onMouseOut={(e) => (e.currentTarget.style.boxShadow = "0 4px 6px rgba(0, 0, 0, 0.1)")} 
        >
          {/* Icono de búsqueda */}
          <FaSearch
            style={{
              marginRight: "10px", 
              color: "#888", 
              fontSize: "1.5rem", 
            }}
          />
          {/* Input de texto para la búsqueda */}
          <input
            type="text"
            placeholder="Search in Reuters..."
            value={query}
            onChange={(e) => setQuery(e.target.value)} // Actualiza el estado de la búsqueda
            style={{
              border: "none", 
              outline: "none", 
              fontSize: "1rem", 
              width: "100%", 
              color: "#333", 
              transition: "box-shadow 0.3s ease, border-color 0.3s ease", 
            }}
            onFocus={(e) => {
              e.target.parentElement.style.boxShadow = `0 8px 20px rgba(0, 123, 255, 0.2)`;
            }}
            onBlur={(e) => {
              e.target.parentElement.style.boxShadow = "0 4px 6px rgba(0, 0, 0, 0.1)"; 
            }}
            onKeyDown={handleKeyDown} 
          />
        </div>

        {/* Selector de método */}
        <div
          style={{
            position: "absolute", 
            top: "-400px", 
            right: "20px", 
          }}
        >
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)} // Actualiza el estado al seleccionar un método
            style={{
              padding: "10px", 
              borderRadius: "5px", 
              border: `1px solid #ddd`, 
              fontSize: "1rem", 
              width: "150px",
              transition: "border-color 0.3s ease, box-shadow 0.3s ease", 
            }}
            onMouseOver={(e) => (e.target.style.borderColor = hoverFocusColor)} 
            onMouseOut={(e) => (e.target.style.borderColor = "#ddd")} 
            onFocus={(e) => {
              e.target.style.boxShadow = `0 0 8px ${hoverFocusColor}`; 
              e.target.style.borderColor = hoverFocusColor; 
            }}
            onBlur={(e) => {
              e.target.style.boxShadow = "none"; 
              e.target.style.borderColor = "#ddd"; 
            }}
          >
            <option value="tfidf">TF-IDF</option> {/* Método TF-IDF */}
            <option value="bow">Bag of Words</option> {/* Método Bag of Words */}
            <option value="word2vec">Word2Vec</option> {/* Método Word2Vec */}
          </select>
        </div>
      </div>
    </div>
  );
}

export default Home;
