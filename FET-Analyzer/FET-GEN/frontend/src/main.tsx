import React from "react";
import ReactDOM from "react-dom/client";
import { lazy, Suspense } from "react";
import "./styles.css";

const App = lazy(() => import("./App"));
const DatabaseLabApp = lazy(() =>
  import("./components/DatabaseLabApp").then((module) => ({
    default: module.DatabaseLabApp
  }))
);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Suspense fallback={<div className="plot-loading"><span className="loading-ring" />Loading application</div>}>
      {window.location.pathname.startsWith("/database-lab") ? <DatabaseLabApp /> : <App />}
    </Suspense>
  </React.StrictMode>
);
