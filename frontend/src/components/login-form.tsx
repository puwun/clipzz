"use client";

import { cn } from "~/lib/utils";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Button } from "./ui/button";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useState } from "react";
import Link from "next/link";
import {
  loginSchema,
  type LoginFormValues,
} from "~/schemas/auth";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";
import { FcGoogle } from "react-icons/fc";


export function LoginForm({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const router = useRouter();

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormValues>({ resolver: zodResolver(loginSchema) });

  const onSubmit = async (data: LoginFormValues) => {
    try {
      setIsSubmitting(true);
      setError(null);

      const signInResult = await signIn("credentials", {
        email: data.email,
        password: data.password,
        redirect: false, // Prevent automatic redirection
      });

      if (signInResult?.error) {
        setError("Invalid email or password.");
      } else if (signInResult?.ok) {
        // Redirect to the dashboard if login is successful
        router.push("/dashboard");
      } else {
        setError("An unexpected error occurred.");
      }
    } catch (error) {
      console.error("Login error:", error);
      setError("An unexpected error occurred.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={cn("flex flex-col gap-6", className)} {...props}>
      <Card>
        <CardHeader>
          <CardTitle className="text-2xl">Login</CardTitle>
          <CardDescription>
            Enter your email below to log in to your account
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit(onSubmit)}>
            <div className="flex flex-col gap-6">
             <Button
                type="button"
                onClick={() => signIn("google")}
                className="w-full bg-white-500 text-gray-800 py-2 rounded-md flex items-center justify-center gap-2 border border-gray-300 hover:text-white transition-color "
              >
                <FcGoogle className="h-5 w-5" />
                Sign in with Google
              </Button>
              <div className="grid gap-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="m@example.com"
                  required
                  {...register("email")}
                />
                {errors.email && (
                  <p className="text-sm text-red-500">{errors.email.message}</p>
                )}
              </div>
              <div className="grid gap-2">
                <div className="flex items-center">
                  <Label htmlFor="password">Password</Label>
                </div>
                <Input
                  id="password"
                  type="password"
                  required
                  {...register("password")}
                />
                {errors.password && (
                  <p className="text-sm text-red-500">
                    {errors.password.message}
                  </p>
                )}
              </div>

              {error && (
                <p className="rounded-md bg-red-50 p-3 text-sm text-red-500">
                  {error}
                </p>
              )}

              <Button type="submit" className="w-full" disabled={isSubmitting}>
                {isSubmitting ? "Logging in..." : "Log in"}
              </Button>
            </div>
            <div className="mt-4 text-center text-sm">
              Don&apos;t have an account?{" "}
              <Link href="/signup" className="underline underline-offset-4">
                Sign up
              </Link>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}