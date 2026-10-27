import SwiftUI

/// The pieces of "Obrazy" that normally live inside the section or a sheet,
/// exposed as plain views.
///
/// `ImagesView` presents all of them itself, so nothing in the app needs this —
/// it exists so each one can be hosted anywhere it makes sense and so the
/// render harness that reviews this module can put it in a window and look at
/// it. They all read `ImagesStore` from the environment, exactly as the section
/// does.
public enum ImagesScreens {

    /// The whole section.
    public static var root: some View { ImagesView() }

    /// The left pane: model, prompt, size, parameters, "Generuj".
    public static var composer: some View { ImageComposer() }

    /// The gallery grid.
    public static var gallery: some View { ImageGallery() }

    /// One image in full, with every parameter it was made with.
    public static func detail(_ image: ImageDTO) -> some View {
        ImageDetailSheet(image: image)
    }
}
