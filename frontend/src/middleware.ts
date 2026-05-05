import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";

const isProtectedRoute = createRouteMatcher(["/chat(.*)", "/admin(.*)"]);

export default clerkMiddleware(async (auth, req) => {
  if (isProtectedRoute(req)) {
    const authData = await auth();
    if (!authData.userId) {
      return authData.redirectToSignIn({ returnBackUrl: req.url });
    }
  }
});

export const config = {
  matcher: ["/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpg|jpeg|png|gif|svg|ttf|woff2?|ico)).*)", "/(api|trpc)(.*)"]
};
