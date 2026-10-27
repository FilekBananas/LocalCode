import Foundation
import LocalCodeKit

/// The `/api/catalog` endpoints, added to the shared `APIClient` actor.
extension APIClient {

    /// `GET /api/catalog?category=&q=&installed=`.
    ///
    /// Needs its own request/response pair rather than `send`, because the
    /// answer's *header* matters: `X-Catalog-Offline: 1` means the server could
    /// not reach Hugging Face and answered from what it already knew, which the
    /// store says out loud above the list.
    public func catalog(category: String? = nil,
                        query text: String? = nil,
                        installedOnly: Bool = false,
                        limit: Int? = nil,
                        minTrust: Int? = nil,
                        officialOnly: Bool = false,
                        sort: String? = nil) async throws -> CatalogPage {
        let path = query("api/catalog", [
            "category": category,
            "q": text,
            "installed": installedOnly ? "1" : nil,
            "limit": limit.map(String.init),
            "minTrust": minTrust.map(String.init),
            "official": officialOnly ? "1" : nil,
            "sort": sort,
        ])
        var request = makeRequest(path)
        // The first call of a session may refresh the catalog from the network.
        request.timeoutInterval = 45
        let (data, headers) = try await sendData(request,
                                                 headers: ["X-Catalog-Offline", "X-Catalog-Total"])
        return CatalogPage(entries: try decodeCatalog(data),
                           isOffline: headers["X-Catalog-Offline"] == "1",
                           total: Int(headers["X-Catalog-Total"] ?? "") ?? 0)
    }

    /// `GET /api/catalog/{id}` — the same entry with every variant filled in,
    /// for the "Szczegóły / Zaawansowane" sheet. The id is a repository id and
    /// contains slashes, so it travels as one escaped path segment.
    public func catalogEntry(id: String) async throws -> CatalogEntry {
        var request = makeRequest("api/catalog/\(pathSegment(id))")
        request.timeoutInterval = 45
        return try await send(request, as: CatalogEntry.self)
    }

    /// `GET /api/catalog/lookup?ref=` — the row for a pasted repository id or
    /// Hugging Face link, whether or not it is part of our catalogue.
    public func catalogLookup(ref: String) async throws -> CatalogEntry {
        var request = makeRequest(query("api/catalog/lookup", ["ref": ref]))
        request.timeoutInterval = 30
        return try await send(request, as: CatalogEntry.self)
    }

    /// `GET /api/logos/{slug}` — the laboratory's mark, as PNG bytes.
    ///
    /// Raw bytes rather than a URL for `AsyncImage`: every endpoint here needs
    /// the bearer token, which `AsyncImage` cannot attach.
    public func publisherLogo(slug: String) async throws -> Data {
        try await sendData(makeRequest("api/logos/\(pathSegment(slug))"))
    }

    /// `POST /api/catalog/install {id}` → the download job it started (201).
    ///
    /// Without `variantId` the server picks the quantisation for this Mac —
    /// that is the point of the store. With one, the person has opened
    /// "Szczegóły" and chosen a different file on purpose.
    @discardableResult
    public func installFromCatalog(id: String, variantId: String? = nil) async throws -> DownloadJob {
        var fields: [String: JSONValue] = ["id": .string(id)]
        if let variantId, !variantId.isEmpty { fields["variantId"] = .string(variantId) }
        let body = try encodeBody(JSONObject(fields))
        return try await send(makeRequest("api/catalog/install", method: "POST", body: body),
                              as: DownloadJob.self)
    }

    // MARK: - Helpers

    private func decodeCatalog(_ data: Data) throws -> [CatalogEntry] {
        do {
            return try JSONDecoder().decode([CatalogEntry].self, from: data)
        } catch {
            throw APIError(message: "Nie udało się odczytać katalogu modeli.",
                           status: nil,
                           code: "decode_error")
        }
    }
}
