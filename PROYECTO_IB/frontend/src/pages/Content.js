import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import reutersLogo from "../img/reuters.png"; // Asegúrate de importar el logo correctamente
import { FaArrowLeft } from "react-icons/fa"; // Para el icono de la flecha

function Content() {
  const location = useLocation();
  const navigate = useNavigate(); // Para manejar la navegación
  const result = location.state; // Recibe los datos del resultado seleccionado

  return (
    <div
      style={{
        fontFamily: "'Arial', sans-serif",
        backgroundColor: "#f8f9fa",
        minHeight: "100vh",
        paddingTop: "75px", // Espacio para el navbar
      }}
    >
      {/* Navbar estático */}
      <nav
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100%",
          height: "65px",
          zIndex: 1000,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between", // Asegura que la flecha y el contenido estén bien distribuidos
          backgroundColor: "#007bff",
          color: "white",
          padding: "0 20px",
          boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
        }}
      >
        {/* Botón de volver */}
        <button
          onClick={() => navigate(-1)} // Navegar hacia atrás
          style={{
            backgroundColor: "transparent",
            border: "none",
            color: "white",
            fontSize: "1.5rem",
            cursor: "pointer",
          }}
        >
          <FaArrowLeft /> {/* Icono de flecha */}
        </button>

        {/* Contenedor centrado para logo y nombre */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            position: "absolute", // Mantén centrado el contenido
            left: "50%",
            transform: "translateX(-50%)",
          }}
        >
          <img
            src={reutersLogo}
            alt="Reuters Logo"
            style={{
              width: "40px",
              height: "40px",
              borderRadius: "50%",
            }}
          />
          <h1 style={{ fontSize: "1.5rem", margin: 0 }}>Reuters Search</h1>
        </div>
      </nav>

      {/* Contenido principal */}
      <div style={{ maxWidth: "800px", margin: "0 auto", padding: "20px" }}>
        <h1 style={{ fontSize: "2rem", marginBottom: "20px" }}>
          {result.Titulo}
        </h1>
        <p
          style={{
            fontSize: "1.1rem",
            lineHeight: "1.6",
            marginBottom: "20px",
          }}
        >
          {result.Contenido}
        </p>
        <p style={{ fontSize: "0.9rem", color: "#666" }}>
          Similitud: {result.Similitud}
        </p>
        <button
          onClick={() => navigate(-1)} // Volver usando React Router
          style={{
            padding: "10px 20px",
            borderRadius: "5px",
            border: "none",
            backgroundColor: "#007bff",
            color: "#fff",
            cursor: "pointer",
            fontSize: "1rem",
          }}
        >
          Volver
        </button>
      </div>
    </div>
  );
}

export default Content;
