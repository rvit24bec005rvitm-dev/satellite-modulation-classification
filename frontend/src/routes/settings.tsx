import { createFileRoute } from "@tanstack/react-router";
import { SatmodApp } from "@/components/satmod-app";
export const Route = createFileRoute("/settings")({
  component: () => <SatmodApp page="settings" />,
});
