import Foundation
import LocalCodeKit

/// The wire types of `/api/catalog` — the in-app model store.
///
/// The store is deliberately *not* the Hugging Face browser with nicer words on
/// top. A catalog row carries everything a card needs — a friendly name, what
/// the model is for, whether it fits in this Mac's memory, and one verdict on
/// each — so the list never makes a request per model, and the repository id,
/// the quantisation and the file list stay behind "Szczegóły".

// MARK: - Entry

/// One model in the store, as `GET /api/catalog` returns it.
public struct CatalogEntry: Decodable, Identifiable, Hashable, Sendable {
    /// The repository id. It contains slashes, so it is percent-escaped when it
    /// goes into a URL path.
    public let id: String
    /// The name a person recognises: "Qwen3 4B Instruct", not the repo id.
    public var name: String
    /// One sentence about what this model is for.
    public var tagline: String
    /// `kodowanie|czat|rozumowanie|obrazy|wizja|embedding`.
    public var category: String
    public var format: String
    /// The quantisation an install would download right now, and everything the
    /// repository offers — so a card can say what you are getting, and the
    /// details sheet can offer the rest.
    public var quant: String?
    public var quants: [String]
    public var sizeBytes: Int64
    /// Pre-formatted by the server: "ok. 2,5 GB".
    public var sizeLabel: String
    public var fit: CatalogFit
    /// Why the server reached that verdict, in a full Polish sentence.
    public var fitDetail: String
    public var speedHint: String
    public var installed: Bool
    /// The library key of the installed copy, when there is one.
    public var installedKey: String?
    public var recommendedVariant: CatalogVariant?
    /// Full on `GET /api/catalog/{id}`, often empty in the list.
    public var variants: [CatalogVariant]
    public var badges: [String]
    public var source: CatalogSource
    /// The same model in another format, for the details sheet.
    public var alternatives: [CatalogSource]
    /// A gated repository: downloading it needs a Hugging Face token.
    public var requiresToken: Bool
    /// Hand-picked rather than found by a search.
    public var curated: Bool
    /// How much is known about this file, and why (0-100, computed server-side).
    public var trust: CatalogTrust
    /// Who published the weights, and whether this repository is theirs.
    public var publisher: CatalogPublisher
    /// Downloads in the last month, when the index knows them.
    public var downloads: Int?
    public var likes: Int?
    /// Total parameters in billions, and how many of them run per token — the
    /// two numbers are different for a mixture-of-experts model, which is the
    /// whole reason both are shown.
    public var paramsB: Double?
    public var activeParamsB: Double?
    /// Ready for the UI: "30 mld, aktywne 3 mld".
    public var paramsLabel: String
    /// True for a repository pasted by hand: installable, but not part of the
    /// catalogue, so it carries no description written for it.
    public var external: Bool
    /// One Polish caveat worth reading before installing ("model bez filtrów").
    public var watchOut: String?

    private enum CodingKeys: String, CodingKey {
        case id, name, tagline, category, format, quant, quants, sizeBytes, sizeLabel, fit, fitDetail
        case speedHint, installed, installedKey, recommendedVariant, variants, badges
        case source, alternatives, requiresToken, curated
        case trust, publisher, downloads, likes, watchOut
        case paramsB, activeParamsB, paramsLabel, external
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = (try? c.decode(String.self, forKey: .name)) ?? id
        tagline = (try? c.decode(String.self, forKey: .tagline)) ?? ""
        category = (try? c.decode(String.self, forKey: .category)) ?? ""
        format = (try? c.decode(String.self, forKey: .format)) ?? ""
        quant = try? c.decodeIfPresent(String.self, forKey: .quant)
        quants = (try? c.decode([String].self, forKey: .quants)) ?? []
        sizeBytes = (try? c.decode(Int64.self, forKey: .sizeBytes)) ?? 0
        sizeLabel = (try? c.decode(String.self, forKey: .sizeLabel)) ?? ""
        fit = (try? c.decode(CatalogFit.self, forKey: .fit)) ?? .unknown
        fitDetail = (try? c.decode(String.self, forKey: .fitDetail)) ?? ""
        speedHint = (try? c.decode(String.self, forKey: .speedHint)) ?? ""
        installed = (try? c.decode(Bool.self, forKey: .installed)) ?? false
        installedKey = try? c.decodeIfPresent(String.self, forKey: .installedKey)
        recommendedVariant = try? c.decodeIfPresent(CatalogVariant.self, forKey: .recommendedVariant)
        variants = (try? c.decode([CatalogVariant].self, forKey: .variants)) ?? []
        badges = (try? c.decode([String].self, forKey: .badges)) ?? []
        source = (try? c.decode(CatalogSource.self, forKey: .source)) ?? CatalogSource()
        alternatives = (try? c.decode([CatalogSource].self, forKey: .alternatives)) ?? []
        requiresToken = (try? c.decode(Bool.self, forKey: .requiresToken)) ?? false
        curated = (try? c.decode(Bool.self, forKey: .curated)) ?? false
        trust = (try? c.decode(CatalogTrust.self, forKey: .trust)) ?? CatalogTrust()
        publisher = (try? c.decode(CatalogPublisher.self, forKey: .publisher)) ?? CatalogPublisher()
        downloads = try? c.decodeIfPresent(Int.self, forKey: .downloads)
        likes = try? c.decodeIfPresent(Int.self, forKey: .likes)
        watchOut = try? c.decodeIfPresent(String.self, forKey: .watchOut)
        paramsB = try? c.decodeIfPresent(Double.self, forKey: .paramsB)
        activeParamsB = try? c.decodeIfPresent(Double.self, forKey: .activeParamsB)
        paramsLabel = (try? c.decode(String.self, forKey: .paramsLabel)) ?? ""
        external = (try? c.decode(Bool.self, forKey: .external)) ?? false
    }

    /// The row from the list, with only the parts `GET /api/catalog/{id}`
    /// legitimately knows better folded in — the full variant list and the
    /// alternative sources.
    ///
    /// The size and the memory verdict deliberately stay as the list reported
    /// them. The detail endpoint recomputes them from whichever variant it
    /// picks, and for at least one repository it picks a file that is not the
    /// model at all (`ggml-org/gpt-oss-20b-GGUF` → the 921 MB EAGLE3 draft head
    /// rather than the 12,1 GB MXFP4 weights), which would otherwise make the
    /// sheet contradict the card the user just clicked.
    public func merging(detail: CatalogEntry) -> CatalogEntry {
        var merged = self
        merged.variants = detail.variants.isEmpty ? variants : detail.variants
        merged.alternatives = detail.alternatives.isEmpty ? alternatives : detail.alternatives
        if recommendedVariant == nil {
            merged.recommendedVariant = Self.likelyVariant(in: merged.variants, forSize: sizeBytes)
        }
        return merged
    }

    /// Which variant the card's size promise actually refers to.
    ///
    /// The server's own `recommended` flag is used when it agrees with that
    /// promise; when it does not, the closest match by size wins, because the
    /// download the user is about to start is the one the card was sized for.
    private static func likelyVariant(in variants: [CatalogVariant],
                                      forSize sizeBytes: Int64) -> CatalogVariant? {
        guard !variants.isEmpty else { return nil }
        let flagged = variants.first(where: \.recommended)
        guard sizeBytes > 0 else { return flagged ?? variants.first }
        func agrees(_ variant: CatalogVariant) -> Bool {
            let delta = abs(variant.sizeBytes - sizeBytes)
            return Double(delta) <= Double(sizeBytes) * 0.1
        }
        if let flagged, agrees(flagged) { return flagged }
        return variants.min { abs($0.sizeBytes - sizeBytes) < abs($1.sizeBytes - sizeBytes) } ?? flagged
    }

    /// True when this variant is the one an install would actually download.
    public func isRecommended(_ variant: CatalogVariant) -> Bool {
        variant.id == recommendedVariant?.id
    }

    /// Everything a details sheet can show about size and shape, in one line.
    public var technicalSummary: String {
        var parts = [format.uppercased(), sizeLabel]
        if !paramsLabel.isEmpty { parts.append(paramsLabel) }
        if let quant = recommendedVariant?.quant, !quant.isEmpty { parts.append(quant) }
        return parts.filter { !$0.isEmpty }.joined(separator: " · ")
    }
}

/// How much LocalCode knows about a file, and why.
///
/// The score is computed by `localcode.hub.trust` from six facts the Hub gives
/// us — who published it, what file formats are inside, the licence, how many
/// people use it and when it was last touched. It is *not* a malware scan, and
/// the app never presents it as one: the reasons travel with the number so a
/// card can always answer "why 62?".
public struct CatalogTrust: Decodable, Hashable, Sendable {
    /// 0-100, or `nil` when the server sent nothing (an older server).
    public var score: Int?
    /// `high|good|medium|low|unknown|rejected` — picks the colour.
    public var level: String
    /// The Polish band name: "wysokie zaufanie", "ostrożnie", "odrzucony"…
    public var label: String
    /// What raises trust, one sentence each.
    public var reasons: [String]
    /// What lowers it — pickle files, no licence, abandoned repository…
    public var warnings: [String]
    /// The licence as a person would name it ("Apache 2.0").
    public var license: String?
    /// Needs a Hugging Face token and an accepted licence.
    public var gated: Bool
    /// True when the model has been rejected (suspicious files or unknown creator).
    public var excluded: Bool
    /// Polish sentence explaining why the model was excluded, when `excluded` is true.
    public var exclusion: String?

    public init() {
        score = nil
        level = "unknown"
        label = ""
        reasons = []
        warnings = []
        license = nil
        gated = false
        excluded = false
        exclusion = nil
    }

    private enum CodingKeys: String, CodingKey {
        case score, level, label, reasons, warnings, license, gated, excluded, exclusion
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        score = try? c.decodeIfPresent(Int.self, forKey: .score)
        level = (try? c.decode(String.self, forKey: .level)) ?? "unknown"
        label = (try? c.decode(String.self, forKey: .label)) ?? ""
        reasons = (try? c.decode([String].self, forKey: .reasons)) ?? []
        warnings = (try? c.decode([String].self, forKey: .warnings)) ?? []
        license = try? c.decodeIfPresent(String.self, forKey: .license)
        gated = (try? c.decode(Bool.self, forKey: .gated)) ?? false
        excluded = (try? c.decode(Bool.self, forKey: .excluded)) ?? false
        exclusion = try? c.decodeIfPresent(String.self, forKey: .exclusion)
    }
}

/// Who stands behind a model: the laboratory that trained it, and — when they
/// differ — whoever converted it to GGUF or MLX.
public struct CatalogPublisher: Decodable, Hashable, Sendable {
    /// `official|trusted|community|unknown`.
    public var kind: String
    /// The laboratory's display name: "Qwen (Alibaba)".
    public var publisher: String
    /// Who packaged it, when that is somebody else ("Unsloth").
    public var packager: String?
    /// The repository the weights were converted from.
    public var upstream: String?
    /// True only for the laboratory's own repository.
    public var official: Bool
    /// The badge text: "źródło oficjalne", "sprawdzona konwersja"…
    public var label: String
    /// One sentence for the tooltip and the details sheet.
    public var detail: String
    /// The laboratory's logo slug (`GET /api/logos/{slug}`), when we ship one.
    public var logo: String?

    public init() {
        kind = "unknown"
        publisher = ""
        packager = nil
        upstream = nil
        official = false
        label = ""
        detail = ""
        logo = nil
    }

    private enum CodingKeys: String, CodingKey {
        case kind, publisher, packager, upstream, official, label, detail, logo
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        kind = (try? c.decode(String.self, forKey: .kind)) ?? "unknown"
        publisher = (try? c.decode(String.self, forKey: .publisher)) ?? ""
        packager = try? c.decodeIfPresent(String.self, forKey: .packager)
        upstream = try? c.decodeIfPresent(String.self, forKey: .upstream)
        official = (try? c.decode(Bool.self, forKey: .official)) ?? false
        label = (try? c.decode(String.self, forKey: .label)) ?? ""
        detail = (try? c.decode(String.self, forKey: .detail)) ?? ""
        logo = try? c.decodeIfPresent(String.self, forKey: .logo)
    }

    /// The symbol next to the score: a seal for the laboratory's own file.
    public var systemImage: String {
        switch kind {
        case "official": "checkmark.seal.fill"
        case "trusted": "shippingbox.fill"
        case "community": "person.2.fill"
        default: "questionmark.circle"
        }
    }
}

/// The memory verdict. The server sends the Polish label itself, so this only
/// has to recognise it and pick a colour.
public enum CatalogFit: String, Decodable, Hashable, Sendable {
    case fits = "Zmieści się"
    case tight = "Na styk"
    case tooBig = "Za duży"
    case unknown = "Nie wiadomo"

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = CatalogFit(rawValue: raw) ?? .unknown
    }

    public var label: String { rawValue }

    public var systemImage: String {
        switch self {
        case .fits: "checkmark.circle.fill"
        case .tight: "exclamationmark.circle.fill"
        case .tooBig: "xmark.circle.fill"
        case .unknown: "questionmark.circle"
        }
    }
}

/// One downloadable shape of a model: a quantisation and the files it needs.
public struct CatalogVariant: Decodable, Identifiable, Hashable, Sendable {
    public let id: String
    public var label: String
    public var sizeBytes: Int64
    public var quant: String?
    public var files: [String]
    public var recommended: Bool
    /// Whether this one quantisation fits this Mac (`nil` from an older server).
    public var fit: CatalogFit?
    public var fitDetail: String

    private enum CodingKeys: String, CodingKey {
        case id, label, sizeBytes, quant, files, recommended, fit, fitDetail
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        label = (try? c.decode(String.self, forKey: .label)) ?? id
        sizeBytes = (try? c.decode(Int64.self, forKey: .sizeBytes)) ?? 0
        quant = try? c.decodeIfPresent(String.self, forKey: .quant)
        files = (try? c.decode([String].self, forKey: .files)) ?? []
        recommended = (try? c.decode(Bool.self, forKey: .recommended)) ?? false
        fit = try? c.decodeIfPresent(CatalogFit.self, forKey: .fit)
        fitDetail = (try? c.decode(String.self, forKey: .fitDetail)) ?? ""
    }

    /// What the quantisation means for the person choosing, in one phrase.
    public var qualityHint: String {
        let q = (quant ?? label).uppercased()
        if q.contains("F16") || q.contains("F32") || q.contains("BF16") { return "pełna precyzja — największa" }
        if q.hasPrefix("Q8") || q.hasPrefix("8BIT") || q.contains("8BIT") { return "8-bit — praktycznie jak oryginał" }
        if q.hasPrefix("Q6") || q.contains("6BIT") { return "6-bit — prawie bez strat" }
        if q.hasPrefix("Q5") || q.contains("5BIT") { return "5-bit — lepsza jakość, trochę większy" }
        if q.hasPrefix("Q4") || q.hasPrefix("IQ4") || q.contains("4BIT") || q.contains("FP4")
            || q.contains("UD-Q4") { return "4-bit — najlepszy kompromis" }
        if q.hasPrefix("Q3") || q.hasPrefix("IQ3") || q.contains("3BIT") || q.contains("UD-Q3") {
            return "3-bit — mały, wyraźnie słabszy"
        }
        if q.hasPrefix("Q2") || q.hasPrefix("IQ2") || q.hasPrefix("IQ1") || q.contains("2BIT")
            || q.contains("UD-Q2") || q.contains("UD-IQ") { return "2-bit — najmniejszy, dużo gorszy" }
        return ""
    }
}

/// Where a model comes from: "Hugging Face" and a repository id.
public struct CatalogSource: Decodable, Hashable, Sendable {
    public var provider: String
    public var repo: String

    public init(provider: String = "", repo: String = "") {
        self.provider = provider
        self.repo = repo
    }

    private enum CodingKeys: String, CodingKey { case provider, repo }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        provider = (try? c.decode(String.self, forKey: .provider)) ?? ""
        repo = (try? c.decode(String.self, forKey: .repo)) ?? ""
    }
}

// MARK: - Settings

/// `GET /api/app-settings` → `catalog`: the categories the store offers and the
/// two sentences it prints around the list. The server owns these strings so a
/// new category never needs an app release.
public struct CatalogAppConfig: Decodable, Hashable, Sendable {
    public var categories: [CatalogCategory]
    /// The orderings the store offers, named by the server.
    public var sorts: [CatalogSortOption]
    /// The quantisation levels a download can be asked for.
    public var qualities: [CatalogSortOption]
    /// "Modele pobierane z Hugging Face".
    public var sourceNote: String
    /// Shown above the list when the catalog could not be refreshed online.
    public var offlineNote: String

    public init() {
        categories = []
        sorts = []
        qualities = []
        sourceNote = ""
        offlineNote = ""
    }

    private enum CodingKeys: String, CodingKey {
        case categories, sorts, qualities, sourceNote, offlineNote
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        categories = (try? c.decode([CatalogCategory].self, forKey: .categories)) ?? []
        sorts = (try? c.decode([CatalogSortOption].self, forKey: .sorts)) ?? []
        qualities = (try? c.decode([CatalogSortOption].self, forKey: .qualities)) ?? []
        sourceNote = (try? c.decode(String.self, forKey: .sourceNote)) ?? ""
        offlineNote = (try? c.decode(String.self, forKey: .offlineNote)) ?? ""
    }
}

/// One entry of the "Sortuj" menu: `{id, label}`.
public struct CatalogSortOption: Decodable, Identifiable, Hashable, Sendable {
    public let id: String
    public var label: String

    public init(id: String, label: String) {
        self.id = id
        self.label = label
    }

    private enum CodingKeys: String, CodingKey { case id, label }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        label = (try? c.decode(String.self, forKey: .label)) ?? id
    }

    /// The icon beside the label in the menu.
    public var systemImage: String {
        switch id {
        case "polecane": "sparkles"
        case "popularne": "arrow.down.circle"
        case "lubiane": "heart"
        case "zaufanie": "checkmark.shield"
        case "oficjalne": "checkmark.seal"
        case "najmniejsze": "arrow.down.right.and.arrow.up.left"
        default: "arrow.up.arrow.down"
        }
    }
}

/// One tab of the store: `{id, label}`.
public struct CatalogCategory: Decodable, Identifiable, Hashable, Sendable {
    public let id: String
    public var label: String

    public init(id: String, label: String) {
        self.id = id
        self.label = label
    }

    private enum CodingKeys: String, CodingKey { case id, label }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        label = (try? c.decode(String.self, forKey: .label)) ?? id
    }

    /// The icon the tab bar draws beside the label.
    public var systemImage: String {
        switch id {
        case "kodowanie": "chevron.left.forwardslash.chevron.right"
        case "czat": "bubble.left.and.bubble.right"
        case "rozumowanie": "brain"
        case "obrazy": "photo"
        case "wizja": "eye"
        case "embedding": "point.3.connected.trianglepath.dotted"
        case "mowa": "waveform"
        case "wideo": "film"
        default: "square.grid.2x2"
        }
    }
}

/// The body of one `GET /api/catalog` call: the rows, plus whether the server
/// had to answer from its own copy (the `X-Catalog-Offline: 1` header).
public struct CatalogPage: Sendable {
    public var entries: [CatalogEntry]
    public var isOffline: Bool
    /// How many models matched before the page limit (`X-Catalog-Total`), so
    /// the store can say "pokazuję 200 z 4312".
    public var total: Int

    public init(entries: [CatalogEntry], isOffline: Bool, total: Int = 0) {
        self.entries = entries
        self.isOffline = isOffline
        self.total = max(total, entries.count)
    }
}
