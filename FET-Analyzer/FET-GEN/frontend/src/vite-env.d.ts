/// <reference types="vite/client" />

import "react";

declare module "react" {
  interface InputHTMLAttributes<T> {
    webkitdirectory?: string | boolean;
    directory?: string | boolean;
  }
}

declare module "plotly.js-basic-dist-min";
