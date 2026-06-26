/// <reference types="vite/client" />

declare module "plotly.js-basic-dist-min";

import "react";

declare module "react" {
  interface InputHTMLAttributes<T> {
    webkitdirectory?: string | boolean;
    directory?: string | boolean;
  }
}

declare module "plotly.js-basic-dist-min";
