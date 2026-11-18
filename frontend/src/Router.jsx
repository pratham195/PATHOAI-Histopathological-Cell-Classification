import { useEffect, useState } from "react";
import App from "./App.jsx";
import ModelEvaluation from "./ModelEvaluation.jsx";

const getPage = () =>
  window.location.hash.startsWith("#/evaluation") ? "evaluation" : "home";

export default function Router() {
  const [page, setPage] = useState(getPage);

  useEffect(() => {
    const onHashChange = () => {
      const next = getPage();
      setPage((prev) => {
        if (prev !== next) window.scrollTo(0, 0);
        return next;
      });
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  return page === "evaluation" ? <ModelEvaluation /> : <App />;
}
