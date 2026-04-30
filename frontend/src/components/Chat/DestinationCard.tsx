import { ExternalLink, MapPin } from "lucide-react";
import type { DestinationMeta } from "../../types";

const FALLBACK_IMAGES: Record<string, string> = {
  Beach: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&q=80",
  Mountain: "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800&q=80",
  City: "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=800&q=80",
  default: "https://images.unsplash.com/photo-1488085061387-422e29b40080?w=800&q=80",
};

interface Props {
  meta: DestinationMeta;
}

export default function DestinationCard({ meta }: Props) {
  const imgSrc = meta.image_url ?? FALLBACK_IMAGES.default;
  const cityName = meta.name.split(",")[0].trim();
  const countryName = meta.name.includes(",") ? meta.name.split(",").slice(1).join(",").trim() : "";

  return (
    <div className="rounded-2xl overflow-hidden bg-[#161b2e] border border-[#1e2540] group
                    hover:border-indigo-500/40 transition-all duration-300 hover:shadow-lg hover:shadow-indigo-500/10">
      {/* Image */}
      <div className="relative h-36 overflow-hidden">
        <img
          src={imgSrc}
          alt={meta.name}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
          onError={(e) => {
            (e.target as HTMLImageElement).src = FALLBACK_IMAGES.default;
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-[#161b2e]/90 via-[#161b2e]/20 to-transparent" />
        <div className="absolute bottom-0 left-0 right-0 px-3 pb-2.5 pt-4">
          <div className="flex items-center gap-1.5">
            <MapPin className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
            <div className="min-w-0">
              <p className="text-white font-semibold text-sm leading-tight truncate">{cityName}</p>
              {countryName && (
                <p className="text-slate-400 text-xs leading-tight truncate">{countryName}</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Footer with source link */}
      {meta.source_url && (
        <div className="px-3 py-2.5 border-t border-[#1e2540]">
          <a
            href={meta.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            <ExternalLink className="w-3.5 h-3.5 shrink-0" />
            <span className="truncate">View travel guide</span>
          </a>
        </div>
      )}
    </div>
  );
}
