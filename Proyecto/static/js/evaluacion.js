document.addEventListener("DOMContentLoaded", function () {
  const button = document.getElementById("toggleButton");
  const metricas = document.getElementById("metricas");

  button.addEventListener("click", function () {
    const visible = metricas.style.display === "block";
    metricas.style.display = visible ? "none" : "block";
    button.textContent = visible ? "Ver métricas de evaluación" : "Ocultar métricas";
  });
});
